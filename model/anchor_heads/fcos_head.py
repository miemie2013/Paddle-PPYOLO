#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import numpy as np
import math
import copy
import paddle
from paddle.fluid.initializer import Normal
from paddle import fluid
import paddle.fluid.layers as L

from model.custom_layers import *


class FCOSHead(paddle.nn.Layer):
    def __init__(self,
                 in_channel,
                 num_classes,
                 fpn_stride=[8, 16, 32, 64, 128],
                 thresh_with_ctr=True,
                 prior_prob=0.01,
                 num_convs=4,
                 norm_type="gn",
                 fcos_loss=None,
                 norm_reg_targets=True,
                 centerness_on_reg=True,
                 use_dcn_in_tower=False,
                 drop_block=False,
                 dcn_v2_stages=[],
                 use_dcn_bias=False,
                 coord_conv=False,
                 spp=False,
                 iou_aware=False,
                 iou_aware_on_reg=True,
                 iou_aware_factor=0.4,
                 nms_cfg=None
                 ):
        super(FCOSHead, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride[::-1]
        self.thresh_with_ctr = thresh_with_ctr
        self.prior_prob = prior_prob
        self.num_convs = num_convs
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg
        self.drop_block = drop_block
        self.norm_type = norm_type
        self.fcos_loss = fcos_loss
        self.dcn_v2_stages = dcn_v2_stages
        self.coord_conv = coord_conv
        self.use_spp = spp
        self.iou_aware = iou_aware
        self.iou_aware_on_reg = iou_aware_on_reg
        self.iou_aware_factor = iou_aware_factor
        self.nms_cfg = nms_cfg


        self.scales_on_reg = paddle.nn.ParameterList()       # 回归分支（预测框坐标）的系数
        self.cls_convs = paddle.nn.LayerList()   # 每个fpn输出特征图  共享的  再进行卷积的卷积层，用于预测类别
        self.reg_convs = paddle.nn.LayerList()   # 每个fpn输出特征图  共享的  再进行卷积的卷积层，用于预测坐标
        # 用于预测centerness
        self.ctn_conv = Conv2dUnit(in_channel, 1, 3, stride=1, bias_attr=True, act=None,
                                   weight_init=Normal(loc=0., scale=0.01), bias_init=Constant(0), name="fcos_head_centerness")

        # 用于预测iou_aware
        self.iaw_conv = None
        if self.iou_aware:
            self.iaw_conv = Conv2dUnit(in_channel, 1, 3, stride=1, bias_attr=True, act=None,
                                       weight_init=Normal(loc=0., scale=0.01), bias_init=Constant(0), name="fcos_head_iou_aware")
        self.coordConv = None
        if self.coord_conv:
            self.coordConv = CoordConv()
        self.spp = None
        if self.use_spp:   # spp层暂定放在第一个卷积层之后
            self.spp_layer = SPP()

        # 每个fpn输出特征图  共享的  卷积层。
        for lvl in range(0, self.num_convs):
            use_dcn = lvl in self.dcn_v2_stages
            bias_attr = True
            if use_dcn:
                bias_attr = use_dcn_bias

            in_ch = self.in_channel
            # if self.coord_conv:
            #     in_ch = self.in_channel + 2 if lvl == 0 else self.in_channel
            in_ch = in_ch * 4 if self.use_spp and lvl == 1 else in_ch   # spp层暂定放在第一个卷积层之后
            cls_conv_layer = Conv2dUnit(in_ch, self.in_channel, 3, stride=1, bias_attr=bias_attr, norm_type=norm_type, groups=32, bias_lr=2.0,
                                        weight_init=Normal(loc=0., scale=0.01), bias_init=Constant(0.0),
                                        act='relu', use_dcn=use_dcn, name='fcos_head_cls_tower_conv_{}'.format(lvl))
            self.cls_convs.append(cls_conv_layer)


            in_ch = self.in_channel
            if self.coord_conv:
                in_ch = self.in_channel + 2 if lvl == 0 else self.in_channel
            in_ch = in_ch * 4 if self.use_spp and lvl == 1 else in_ch   # spp层暂定放在第一个卷积层之后
            reg_conv_layer = Conv2dUnit(in_ch, self.in_channel, 3, stride=1, bias_attr=bias_attr, norm_type=norm_type, groups=32, bias_lr=2.0,
                                        weight_init=Normal(loc=0., scale=0.01), bias_init=Constant(0.0),
                                        act='relu', use_dcn=use_dcn, name='fcos_head_reg_tower_conv_{}'.format(lvl))
            self.reg_convs.append(reg_conv_layer)

        self.drop_block1 = None
        self.drop_block2 = None
        if self.drop_block:
            self.drop_block1 = DropBlock(block_size=3, keep_prob=0.9, is_test=False)
            self.drop_block2 = DropBlock(block_size=3, keep_prob=0.9, is_test=False)

        # 类别分支最后的卷积。设置偏移的初始值使得各类别预测概率初始值为self.prior_prob (根据激活函数是sigmoid()时推导出，和RetinaNet中一样)
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        cls_last_conv_layer = Conv2dUnit(in_channel, self.num_classes, 3, stride=1, bias_attr=True, act=None,
                                         weight_init=Normal(loc=0., scale=0.01), bias_init=Constant(bias_init_value), name="fcos_head_cls")
        # 坐标分支最后的卷积
        reg_last_conv_layer = Conv2dUnit(in_channel, 4, 3, stride=1, bias_attr=True, act=None,
                                         weight_init=Normal(loc=0., scale=0.01), bias_init=Constant(0), name="fcos_head_reg")
        self.cls_convs.append(cls_last_conv_layer)
        self.reg_convs.append(reg_last_conv_layer)


        n = len(self.fpn_stride)      # 有n个输出层
        for i in range(n):     # 遍历每个输出层
            scale = fluid.layers.create_parameter(shape=[1, ], dtype='float32',
                                                  attr=ParamAttr(name="scale_%d" % i, learning_rate=1.0, initializer=Constant(1.0)),
                                                  default_initializer=Constant(1.0))
            self.scales_on_reg.append(scale)

        self.relu = paddle.nn.ReLU()

    def set_dropblock(self, is_test):
        if self.drop_block:
            self.drop_block1.is_test = is_test
            self.drop_block2.is_test = is_test

    def _fcos_head(self, features, fpn_stride, i, is_training=False):
        """
        Args:
            features (Variables): feature map from FPN
            fpn_stride     (int): the stride of current feature map
            is_training   (bool): whether is train or test mode
        """
        fpn_scale = self.scales_on_reg[i]
        subnet_blob_cls = features
        subnet_blob_reg = features
        if self.coord_conv:
            # subnet_blob_cls = self.coordConv(subnet_blob_cls)
            subnet_blob_reg = self.coordConv(subnet_blob_reg)
        for lvl in range(0, self.num_convs):
            subnet_blob_cls = self.cls_convs[lvl](subnet_blob_cls)
            subnet_blob_reg = self.reg_convs[lvl](subnet_blob_reg)
            if self.use_spp and lvl == 0:   # spp层暂定放在第一个卷积层之后
                subnet_blob_cls = self.spp_layer(subnet_blob_cls)
                subnet_blob_reg = self.spp_layer(subnet_blob_reg)

        if self.drop_block:
            subnet_blob_cls = self.drop_block1(subnet_blob_cls)
            subnet_blob_reg = self.drop_block2(subnet_blob_reg)

        cls_logits = self.cls_convs[self.num_convs](subnet_blob_cls)   # 通道数变成类别数
        bbox_reg = self.reg_convs[self.num_convs](subnet_blob_reg)     # 通道数变成4
        bbox_reg = bbox_reg * fpn_scale     # 预测坐标的特征图整体乘上fpn_scale，是一个可学习参数
        # 如果 归一化坐标分支，bbox_reg进行relu激活
        if self.norm_reg_targets:
            bbox_reg = self.relu(bbox_reg)
            if not is_training:   # 验证状态的话，bbox_reg再乘以下采样倍率
                bbox_reg = bbox_reg * fpn_stride
        else:
            bbox_reg = L.exp(bbox_reg)


        # ============= centerness分支，默认是用坐标分支接4个卷积层之后的结果subnet_blob_reg =============
        if self.centerness_on_reg:
            centerness = self.ctn_conv(subnet_blob_reg)
        else:
            centerness = self.ctn_conv(subnet_blob_cls)

        # ============= iou_aware分支，默认是用坐标分支接4个卷积层之后的结果subnet_blob_reg =============
        iou_aware = None
        if self.iou_aware:
            if self.iou_aware_on_reg:
                iou_aware = self.iaw_conv(subnet_blob_reg)
            else:
                iou_aware = self.iaw_conv(subnet_blob_cls)
        return cls_logits, bbox_reg, centerness, iou_aware

    def _get_output(self, body_feats, is_training=False):
        """
        Args:
            body_feates (list): the list of fpn feature maps。[p7, p6, p5, p4, p3]
            is_training (bool): whether is train or test mode
        Return:
            cls_logits (Variables): prediction for classification
            bboxes_reg (Variables): prediction for bounding box
            centerness (Variables): prediction for ceterness
        """
        cls_logits = []
        bboxes_reg = []
        centerness = []
        iou_awares = []
        assert len(body_feats) == len(self.fpn_stride), \
            "The size of body_feats is not equal to size of fpn_stride"
        i = 0
        for features, fpn_stride in zip(body_feats, self.fpn_stride):
            cls_pred, bbox_pred, ctn_pred, iaw_pred = self._fcos_head(features, fpn_stride, i, is_training=is_training)
            cls_logits.append(cls_pred)
            bboxes_reg.append(bbox_pred)
            centerness.append(ctn_pred)
            iou_awares.append(iaw_pred)
            i += 1
        return cls_logits, bboxes_reg, centerness, iou_awares

    def _compute_locations(self, features):
        """
        Args:
            features (list): List of Variables for FPN feature maps. [p7, p6, p5, p4, p3]
        Return:
            Anchor points for each feature map pixel
        """
        locations = []
        for lvl, feature in enumerate(features):
            shape_fm = fluid.layers.shape(feature)
            shape_fm.stop_gradient = True
            h = shape_fm[2]
            w = shape_fm[3]
            fpn_stride = self.fpn_stride[lvl]
            shift_x = fluid.layers.range(
                0, w * fpn_stride, fpn_stride, dtype='float32')
            shift_y = fluid.layers.range(
                0, h * fpn_stride, fpn_stride, dtype='float32')
            shift_x = fluid.layers.unsqueeze(shift_x, axes=[0])
            shift_y = fluid.layers.unsqueeze(shift_y, axes=[1])
            shift_x = fluid.layers.expand_as(
                shift_x, target_tensor=feature[0, 0, :, :])
            shift_y = fluid.layers.expand_as(
                shift_y, target_tensor=feature[0, 0, :, :])
            shift_x.stop_gradient = True
            shift_y.stop_gradient = True
            shift_x = fluid.layers.reshape(shift_x, shape=[-1])
            shift_y = fluid.layers.reshape(shift_y, shape=[-1])
            location = fluid.layers.stack(
                [shift_x, shift_y], axis=-1) + fpn_stride // 2
            location.stop_gradient = True
            locations.append(location)
        return locations

    def _postprocessing_by_level(self, locations, box_cls, box_reg, box_ctn, box_iaw,
                                 im_info):
        """
        Args:
            locations (Variables): anchor points for current layer
            box_cls   (Variables): categories prediction
            box_reg   (Variables): bounding box prediction
            box_ctn   (Variables): centerness prediction
            im_info   (Variables): [h, w, scale] for input images
        Return:
            box_cls_ch_last  (Variables): score for each category, in [N, C, M]
                C is the number of classes and M is the number of anchor points
            box_reg_decoding (Variables): decoded bounding box, in [N, M, 4]
                last dimension is [x1, y1, x2, y2]
        """
        batch_size = box_cls.shape[0]
        num_classes = self.num_classes

        # =========== 类别概率，[N, 80, H*W] ===========
        box_cls_ch_last = L.reshape(box_cls, (batch_size, num_classes, -1))  # [N, 80, H*W]
        box_cls_ch_last = L.sigmoid(box_cls_ch_last)  # 类别概率用sigmoid()激活，[N, 80, H*W]

        # =========== 坐标(4个偏移)，[N, H*W, 4] ===========
        box_reg_ch_last = L.transpose(box_reg, perm=[0, 2, 3, 1])  # [N, H, W, 4]
        box_reg_ch_last = L.reshape(box_reg_ch_last, (batch_size, -1, 4))  # [N, H*W, 4]，坐标不用再接激活层，直接预测。

        # =========== centerness，[N, 1, H*W] ===========
        box_ctn_ch_last = L.reshape(box_ctn, (batch_size, 1, -1))  # [N, 1, H*W]
        box_ctn_ch_last = L.sigmoid(box_ctn_ch_last)  # centerness用sigmoid()激活，[N, 1, H*W]

        # =========== iou_aware，[N, 1, H*W] ===========
        if self.iou_aware:
            box_iaw_ch_last = L.reshape(box_iaw, (batch_size, 1, -1))  # [N, 1, H*W]
            box_iaw_ch_last = L.sigmoid(box_iaw_ch_last)  # iou_aware用sigmoid()激活，[N, 1, H*W]

        box_reg_decoding = L.concat(  # [N, H*W, 4]
            [
                locations[:, 0:1] - box_reg_ch_last[:, :, 0:1],  # 左上角x坐标
                locations[:, 1:2] - box_reg_ch_last[:, :, 1:2],  # 左上角y坐标
                locations[:, 0:1] + box_reg_ch_last[:, :, 2:3],  # 右下角x坐标
                locations[:, 1:2] + box_reg_ch_last[:, :, 3:4]  # 右下角y坐标
            ],
            axis=-1)
        # # recover the location to original image
        im_scale = im_info[:, 2]  # [N, ]
        im_scale = L.reshape(im_scale, (batch_size, 1, 1))  # [N, 1, 1]
        box_reg_decoding = box_reg_decoding / im_scale  # [N, H*W, 4]，最终坐标=坐标*图片缩放因子
        if self.thresh_with_ctr:
            box_cls_ch_last = box_cls_ch_last * box_ctn_ch_last  # [N, 80, H*W]，最终分数=类别概率*centerness
        if self.iou_aware:
            # pow运算太慢，所以直接乘。
            # box_cls_ch_last = L.pow(box_cls_ch_last, (1 - self.iou_aware_factor)) \
            #                   * L.pow(box_iaw_ch_last, self.iou_aware_factor)
            box_cls_ch_last = box_cls_ch_last * box_iaw_ch_last
        return box_cls_ch_last, box_reg_decoding

    def _post_processing(self, locations, cls_logits, bboxes_reg, centerness, iou_awares,
                         im_info):
        """
        Args:
            locations   (list): List of Variables composed by center of each anchor point
            cls_logits  (list): List of Variables for class prediction
            bboxes_reg  (list): List of Variables for bounding box prediction
            centerness  (list): List of Variables for centerness prediction
            im_info(Variables): [h, w, scale] for input images
        Return:
            pred (LoDTensor): predicted bounding box after nms,
                the shape is n x 6, last dimension is [label, score, xmin, ymin, xmax, ymax]
        """
        pred_boxes_ = []
        pred_scores_ = []
        for _, (
                pts, cls, box, ctn, iaw
        ) in enumerate(zip(locations, cls_logits, bboxes_reg, centerness, iou_awares)):
            pred_scores_lvl, pred_boxes_lvl = self._postprocessing_by_level(
                pts, cls, box, ctn, iaw, im_info)
            pred_boxes_.append(pred_boxes_lvl)     # [N, H*W, 4]，最终坐标
            pred_scores_.append(pred_scores_lvl)   # [N, 80, H*W]，最终分数
        pred_boxes = L.concat(pred_boxes_, axis=1)    # [N, 所有格子, 4]，最终坐标
        pred_scores = L.concat(pred_scores_, axis=2)  # [N, 80, 所有格子]，最终分数

        # nms
        preds = []
        nms_cfg = copy.deepcopy(self.nms_cfg)
        nms_type = nms_cfg.pop('nms_type')
        if nms_type == 'matrix_nms':
            batch_size = pred_boxes.shape[0]
            for i in range(batch_size):
                pred = fluid.layers.matrix_nms(pred_boxes[i:i+1, :, :], pred_scores[i:i+1, :, :], background_label=-1, **nms_cfg)
                preds.append(pred)
        elif nms_type == 'multiclass_nms':
            batch_size = pred_boxes.shape[0]
            for i in range(batch_size):
                pred = fluid.layers.multiclass_nms(pred_boxes[i:i+1, :, :], pred_scores[i:i+1, :, :], background_label=-1, **nms_cfg)
                preds.append(pred)
        return preds

    def get_loss(self, input, tag_labels, tag_bboxes, tag_centerness):
        """
        Calculate the loss for FCOS
        Args:
            input           (list): List of Variables for feature maps from FPN layers
            tag_labels     (Variables): category targets for each anchor point
            tag_bboxes     (Variables): bounding boxes  targets for positive samples
            tag_centerness (Variables): centerness targets for positive samples
        Return:
            loss (dict): loss composed by classification loss, bounding box
                regression loss and centerness regression loss
        """
        # cls_logits里面每个元素是[N, 80, 格子行数, 格子列数]
        # bboxes_reg里面每个元素是[N,  4, 格子行数, 格子列数]
        # centerness里面每个元素是[N,  1, 格子行数, 格子列数]
        # is_training=True表示训练状态。
        # 训练状态的话，bbox_reg不会乘以下采样倍率，这样得到的坐标单位1表示当前层的1个格子边长。
        # 因为在Gt2FCOSTarget中设置了norm_reg_targets=True对回归的lrtb进行了归一化，归一化方式是除以格子边长（即下采样倍率），
        # 所以网络预测的lrtb的单位1实际上代表了当前层的1个格子边长。
        cls_logits, bboxes_reg, centerness, iou_awares = self._get_output(
            input, is_training=True)
        loss = self.fcos_loss(cls_logits, bboxes_reg, centerness, iou_awares, tag_labels,
                              tag_bboxes, tag_centerness)
        return loss

    def get_prediction(self, input, im_info):
        """
        Decode the prediction
        Args:
            input: [p7, p6, p5, p4, p3]
            im_info(Variables): [h, w, scale] for input images
        Return:
            the bounding box prediction
        """
        # cls_logits里面每个元素是[N, 80, 格子行数, 格子列数]
        # bboxes_reg里面每个元素是[N,  4, 格子行数, 格子列数]
        # centerness里面每个元素是[N,  1, 格子行数, 格子列数]
        # is_training=False表示验证状态。
        # 验证状态的话，bbox_reg再乘以下采样倍率，这样得到的坐标是相对于输入图片宽高的坐标。
        cls_logits, bboxes_reg, centerness, iou_awares = self._get_output(
            input, is_training=False)

        # locations里面每个元素是[格子行数*格子列数, 2]。即格子中心点相对于输入图片宽高的xy坐标。
        locations = self._compute_locations(input)

        preds = self._post_processing(locations, cls_logits, bboxes_reg,
                                     centerness, iou_awares, im_info)
        return preds





