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
from model.matrix_nms import *


class RepPointsHead(paddle.nn.Layer):
    def __init__(self,
                 in_channel,
                 feat_channels=256,
                 point_feat_channels=256,
                 num_classes=80,
                 num_points=9,
                 gradient_mul=0.1,
                 point_base_scale=4,
                 fpn_stride=[8, 16, 32, 64, 128],
                 thresh_with_ctr=True,
                 prior_prob=0.01,
                 num_convs=3,
                 use_grid_points=False,
                 center_init=True,
                 transform_method='moment',
                 moment_mul=0.01,
                 norm_type="gn",
                 reppoints_loss=None,
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
        super(RepPointsHead, self).__init__()
        self.in_channel = in_channel
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.num_classes = num_classes
        self.num_points = num_points
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.fpn_stride = fpn_stride
        self.thresh_with_ctr = thresh_with_ctr
        self.prior_prob = prior_prob
        self.num_convs = num_convs
        self.use_grid_points = use_grid_points
        self.center_init = center_init
        self.transform_method = transform_method
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg
        self.drop_block = drop_block
        self.norm_type = norm_type
        self.reppoints_loss = reppoints_loss
        self.dcn_v2_stages = dcn_v2_stages
        self.coord_conv = coord_conv
        self.use_spp = spp
        self.iou_aware = iou_aware
        self.iou_aware_on_reg = iou_aware_on_reg
        self.iou_aware_factor = iou_aware_factor
        self.nms_cfg = nms_cfg
        self.use_sigmoid_cls = True


        self.cls_convs = paddle.nn.LayerList()   # 每个fpn输出特征图  共享的  再进行卷积的卷积层，用于预测类别
        self.reg_convs = paddle.nn.LayerList()   # 每个fpn输出特征图  共享的  再进行卷积的卷积层，用于预测坐标

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
                                        act='relu', use_dcn=use_dcn, name='reppoints_head_cls_tower_conv_{}'.format(lvl))
            self.cls_convs.append(cls_conv_layer)


            in_ch = self.in_channel
            if self.coord_conv:
                in_ch = self.in_channel + 2 if lvl == 0 else self.in_channel
            in_ch = in_ch * 4 if self.use_spp and lvl == 1 else in_ch   # spp层暂定放在第一个卷积层之后
            reg_conv_layer = Conv2dUnit(in_ch, self.in_channel, 3, stride=1, bias_attr=bias_attr, norm_type=norm_type, groups=32, bias_lr=2.0,
                                        weight_init=Normal(loc=0., scale=0.01), bias_init=Constant(0.0),
                                        act='relu', use_dcn=use_dcn, name='reppoints_head_reg_tower_conv_{}'.format(lvl))
            self.reg_convs.append(reg_conv_layer)

        self.drop_block1 = None
        self.drop_block2 = None
        if self.drop_block:
            self.drop_block1 = DropBlock(block_size=3, keep_prob=0.9, is_test=False)
            self.drop_block2 = DropBlock(block_size=3, keep_prob=0.9, is_test=False)

        # 类别分支最后2个卷积。设置偏移的初始值使得各类别预测概率初始值为self.prior_prob (根据激活函数是sigmoid()时推导出，和RetinaNet中一样)
        self.reppoints_cls_conv_w = self.create_parameter(
            shape=[self.point_feat_channels, in_channel, 3, 3], dtype='float32',
            attr=ParamAttr(name="reppoints_cls_conv_w", learning_rate=1.0, initializer=Normal(loc=0., scale=0.01)),
            default_initializer=fluid.initializer.Xavier())
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.reppoints_cls_out = Conv2dUnit(self.point_feat_channels, self.num_classes, 1, stride=1, bias_attr=True, act=None,
                                            weight_init=Normal(loc=0., scale=0.01), bias_init=Constant(bias_init_value),
                                            name="reppoints_cls_out")


        # 点分支最后4个卷积
        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points
        self.reppoints_pts_init_conv = Conv2dUnit(in_channel, self.point_feat_channels, 3, stride=1, bias_attr=True, act=None,
                                                  weight_init=Normal(loc=0., scale=0.01), bias_init=Constant(0),
                                                  name="reppoints_pts_init_conv")
        self.reppoints_pts_init_out = Conv2dUnit(self.point_feat_channels, pts_out_dim, 1, stride=1, bias_attr=True, act=None,
                                                 weight_init=Normal(loc=0., scale=0.01), bias_init=Constant(0),
                                                 name="reppoints_pts_init_out")

        self.reppoints_pts_refine_conv_w = self.create_parameter(
            shape=[self.point_feat_channels, in_channel, 3, 3], dtype='float32',
            attr=ParamAttr(name="reppoints_pts_refine_conv_w", learning_rate=1.0, initializer=Normal(loc=0., scale=0.01)),
            default_initializer=fluid.initializer.Xavier())
        self.reppoints_pts_refine_out = Conv2dUnit(self.point_feat_channels, pts_out_dim, 1, stride=1, bias_attr=True, act=None,
                                                   weight_init=Normal(loc=0., scale=0.01), bias_init=Constant(0),
                                                   name="reppoints_pts_refine_out")

        # 全局学习的系数
        if self.transform_method == 'moment':
            self.moment_transfer = self.create_parameter(shape=[2, ], dtype='float32',
                                                         attr=ParamAttr(name="moment_transfer", learning_rate=1.0, initializer=Constant(0.0)),
                                                         default_initializer=fluid.initializer.Xavier())
            self.moment_mul = moment_mul
        self.relu = paddle.nn.ReLU()


        self.point_generators = [PointGenerator() for _ in self.fpn_stride]


        # we use deform conv to extract points features
        # 我们用可变形卷积来提取点特征。
        self.dcn_kernel = int(np.sqrt(num_points))   # 3=根号9
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)   # 1
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)   # [-1, 0, 1]
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)   # [-1, -1, -1, 0, 0, 0, 1, 1, 1] 第0行第0列的y坐标，第0行第1列的y坐标，...
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)     # [-1, 0, 1, -1, 0, 1, -1, 0, 1] 第0行第0列的x坐标，第0行第1列的x坐标，...

        # [-1, -1, -1, 0, ...]  第0行第0列的yx坐标，第0行第1列的yx坐标，...
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1, ))  # shape=(18, )  9个点的yx坐标
        self.dcn_base_offset = paddle.to_tensor(dcn_base_offset)               # shape=(18, )  9个点的yx坐标
        self.dcn_base_offset = L.reshape(self.dcn_base_offset, (1, -1, 1, 1))  # shape=(1, 18, 1, 1)  9个点的yx坐标
        self.dcn_base_offset.stop_gradient = True


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
        elif nms_type == 'no_nms':
            batch_size = pred_boxes.shape[0]
            for i in range(batch_size):
                pred = no_nms(pred_boxes[i, :, :], pred_scores[i, :, :], **nms_cfg)
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

    def points2bbox(self, pts, y_first=True):
        """点集转换成包围框.

        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_first=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = L.reshape(pts, (pts.shape[0], -1, 2, pts.shape[2], pts.shape[3]))
        pts_y = pts_reshape[:, :, 0, :, :] if y_first else pts_reshape[:, :, 1, :, :]
        pts_x = pts_reshape[:, :, 1, :, :] if y_first else pts_reshape[:, :, 0, :, :]
        if self.transform_method == 'minmax':
            # bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            # bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            # bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            # bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            # bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
            #                  dim=1)
            pass
        elif self.transform_method == 'partial_minmax':
            # pts_y = pts_y[:, :4, ...]
            # pts_x = pts_x[:, :4, ...]
            # bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            # bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            # bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            # bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            # bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
            #                  dim=1)
            pass
        elif self.transform_method == 'moment':
            pts_y_mean = L.reduce_mean(pts_y, dim=1, keep_dim=True)
            pts_x_mean = L.reduce_mean(pts_x, dim=1, keep_dim=True)
            pts_y_std = paddle.std(pts_y - pts_y_mean, axis=1, keepdim=True)
            pts_x_std = paddle.std(pts_x - pts_x_mean, axis=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * L.exp(moment_width_transfer)
            half_height = pts_y_std * L.exp(moment_height_transfer)
            bbox = L.concat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ], axis=1)
        else:
            raise NotImplementedError
        return bbox

    def get_prediction(self, input, img_metas, rescale=False, with_nms=True):
        """
        Decode the prediction
        Args:
            input: [p7, p6, p5, p4, p3]
            im_info(Variables): [h, w, scale] for input images
        Return:
            the bounding box prediction
        """
        # cls_scores       里面每个元素是[N, 80, 格子行数, 格子列数]
        # pts_preds_init   里面每个元素是[N, 18, 格子行数, 格子列数]
        # pts_preds_refine 里面每个元素是[N, 18, 格子行数, 格子列数]
        cls_scores = []
        pts_preds_init = []
        pts_preds_refine = []
        for x in input:
            cls_out, pts_out_init, pts_out_refine = self.forward_single(x)
            cls_scores.append(cls_out)
            pts_preds_init.append(pts_out_init)
            pts_preds_refine.append(pts_out_refine)

        # bbox_preds_refine 里面每个元素是[N,  4, 格子行数, 格子列数]
        bbox_preds_refine = [
            self.points2bbox(pts_pred_refine)
            for pts_pred_refine in pts_preds_refine
        ]

        num_levels = len(cls_scores)
        # mlvl_points 里面每个元素是[格子行数*格子列数, 3]  具体是(格子左上角x坐标, 格子左上角y坐标, 格子边长)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].shape[-2:], self.fpn_stride[i])
            for i in range(num_levels)
        ]
        result_list = []

        # 遍历这批每一张图片
        n = len(img_metas)
        for img_id in range(n):
            # 这张图片的分数特征图, [80, s8, s8], [80, s16, s16], ...
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            # 这张图片的预测框特征图, [4, s8, s8], [4, s16, s16], ...
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, rescale,
                                                with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           rescale=False,
                           with_nms=True):
        # mlvl_points 里面每个元素是[格子行数*格子列数, 3]  具体是(格子左上角x坐标, 格子左上角y坐标, 格子边长)
        nms_cfg = self.nms_cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        # 遍历每个fpn输出层
        for i_lvl, (cls_score, bbox_pred, points) in enumerate(
                zip(cls_scores, bbox_preds, mlvl_points)):
            # cls_score.shape = [80, h, w]
            # bbox_pred.shape = [ 4, h, w]
            # points.shape    = [h*w, 3]   具体是(格子左上角x坐标, 格子左上角y坐标, 格子边长)
            cls_score = L.transpose(cls_score, [1, 2, 0])              # [h, w, 80]
            cls_score = L.reshape(cls_score, (-1, self.num_classes))   # [h*w, 80]
            if self.use_sigmoid_cls:
                scores = L.sigmoid(cls_score)   # [h*w, 80]
            else:
                scores = L.softmax(cls_score)
            bbox_pred = L.transpose(bbox_pred, [1, 2, 0])   # [h, w, 4]
            bbox_pred = L.reshape(bbox_pred, (-1, 4))       # [h*w, 4]
            nms_top_k = nms_cfg.get('nms_top_k', -1)
            if nms_top_k > 0 and scores.shape[0] > nms_top_k:
                if self.use_sigmoid_cls:
                    max_scores = L.reduce_max(scores, dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    # max_scores, _ = scores[:, :-1].max(dim=1)
                    pass
                _, topk_inds = L.topk(max_scores, k=nms_top_k)
                scores = L.gather(scores, topk_inds)  # [M, 80]
                points = L.gather(points, topk_inds)  # [M, 3]   格子xy坐标、边长
                bbox_pred = L.gather(bbox_pred, topk_inds)  # [M, 4]

            # [M, 4]  格子xy坐标重复2次。格子左上角坐标。
            bbox_pos_center = L.concat([points[:, :2], points[:, :2]], axis=1)

            # [M, 4]  物体最终预测坐标(x1y1x2y2格式) = bbox_pred*格子边长 + 格子左上角坐标
            bboxes = bbox_pred * self.fpn_stride[i_lvl] + bbox_pos_center

            x1 = L.clip(bboxes[:, 0], 0.0, img_shape[1])
            y1 = L.clip(bboxes[:, 1], 0.0, img_shape[0])
            x2 = L.clip(bboxes[:, 2], 0.0, img_shape[1])
            y2 = L.clip(bboxes[:, 3], 0.0, img_shape[0])
            bboxes = paddle.stack([x1, y1, x2, y2], axis=-1)  # [M, 4]
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_scores = L.concat(mlvl_scores, axis=0)  # [M2, 80]  各个fpn层预测的分数汇合在一起
        mlvl_bboxes = L.concat(mlvl_bboxes, axis=0)  # [M2, 4]   各个fpn层预测的bbox(x1y1x2y2格式)汇合在一起
        if rescale:
            scale_factor_ = paddle.to_tensor(scale_factor)
            mlvl_bboxes /= scale_factor_  # [M2, 4]   预测的bbox(x1y1x2y2格式)

        pred_scores = L.unsqueeze(mlvl_scores, axes=0)  # [1, M2, 80]
        pred_boxes = L.unsqueeze(mlvl_bboxes, axes=0)   # [1, M2,  4]，最终坐标
        pred_scores = L.transpose(pred_scores, perm=[0, 2, 1])  # [1, 80, M2]，最终分数

        # nms
        pred = None
        i = 0
        nms_cfg = copy.deepcopy(self.nms_cfg)
        nms_type = nms_cfg.pop('nms_type')
        if nms_type == 'matrix_nms':
            pred = fluid.layers.matrix_nms(pred_boxes[i:i+1, :, :], pred_scores[i:i+1, :, :], background_label=-1, **nms_cfg)
        elif nms_type == 'multiclass_nms':
            pred = fluid.layers.multiclass_nms(pred_boxes[i:i+1, :, :], pred_scores[i:i+1, :, :], background_label=-1, **nms_cfg)
        return pred


    def forward_single(self, x):
        """Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale,
                                      scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)



        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))


        if self.use_grid_points:
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(
                pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset



        mask1 = paddle.ones((dcn_offset.shape[0], dcn_offset.shape[1]//2, dcn_offset.shape[2], dcn_offset.shape[3]), 'float32')
        mask1.stop_gradient = True
        temp1 = deformable_conv(input=cls_feat, offset=dcn_offset, mask=mask1,
                                num_filters=self.point_feat_channels, filter_size=3,
                                stride=1, padding=1, groups=1, deformable_groups=1, im2col_step=1,
                                filter_param=self.reppoints_cls_conv_w, bias_attr=False)
        cls_out = self.reppoints_cls_out(self.relu(temp1))


        temp2 = deformable_conv(input=pts_feat, offset=dcn_offset, mask=mask1,
                                num_filters=self.point_feat_channels, filter_size=3,
                                stride=1, padding=1, groups=1, deformable_groups=1, im2col_step=1,
                                filter_param=self.reppoints_pts_refine_conv_w, bias_attr=False)
        pts_out_refine = self.reppoints_pts_refine_out(self.relu(temp2))


        if self.use_grid_points:
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
                pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()

        return cls_out, pts_out_init, pts_out_refine





