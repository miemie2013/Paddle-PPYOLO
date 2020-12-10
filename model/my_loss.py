#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-23 09:13:23
#   Description : paddle2.0_ppyolo
#
# ================================================================
import paddle
import paddle.nn.functional as F
import paddle.fluid.layers as L
import numpy as np
import paddle.fluid as fluid
import math

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence



def bbox_ciou(boxes1, boxes2):
    '''
    计算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:
    '''

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = L.concat([boxes1[:, :, :, :, :2] - boxes1[:, :, :, :, 2:] * 0.5,
                                boxes1[:, :, :, :, :2] + boxes1[:, :, :, :, 2:] * 0.5], axis=-1)
    boxes2_x0y0x1y1 = L.concat([boxes2[:, :, :, :, :2] - boxes2[:, :, :, :, 2:] * 0.5,
                                boxes2[:, :, :, :, :2] + boxes2[:, :, :, :, 2:] * 0.5], axis=-1)
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = L.concat([L.elementwise_min(boxes1_x0y0x1y1[:, :, :, :, :2], boxes1_x0y0x1y1[:, :, :, :, 2:]),
                                L.elementwise_max(boxes1_x0y0x1y1[:, :, :, :, :2], boxes1_x0y0x1y1[:, :, :, :, 2:])],
                               axis=-1)
    boxes2_x0y0x1y1 = L.concat([L.elementwise_min(boxes2_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, 2:]),
                                L.elementwise_max(boxes2_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, 2:])],
                               axis=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[:, :, :, :, 2] - boxes1_x0y0x1y1[:, :, :, :, 0]) * (
                boxes1_x0y0x1y1[:, :, :, :, 3] - boxes1_x0y0x1y1[:, :, :, :, 1])
    boxes2_area = (boxes2_x0y0x1y1[:, :, :, :, 2] - boxes2_x0y0x1y1[:, :, :, :, 0]) * (
                boxes2_x0y0x1y1[:, :, :, :, 3] - boxes2_x0y0x1y1[:, :, :, :, 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = L.elementwise_max(boxes1_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, :2])
    right_down = L.elementwise_min(boxes1_x0y0x1y1[:, :, :, :, 2:], boxes2_x0y0x1y1[:, :, :, :, 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = L.relu(right_down - left_up)
    inter_area = inter_section[:, :, :, :, 0] * inter_section[:, :, :, :, 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = L.elementwise_min(boxes1_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, :2])
    enclose_right_down = L.elementwise_max(boxes1_x0y0x1y1[:, :, :, :, 2:], boxes2_x0y0x1y1[:, :, :, :, 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = L.pow(enclose_wh[:, :, :, :, 0], 2) + L.pow(enclose_wh[:, :, :, :, 1], 2)

    # 两矩形中心点距离的平方
    p2 = L.pow(boxes1[:, :, :, :, 0] - boxes2[:, :, :, :, 0], 2) + L.pow(boxes1[:, :, :, :, 1] - boxes2[:, :, :, :, 1],
                                                                         2)

    # 增加av。
    atan1 = L.atan(boxes1[:, :, :, :, 2] / (boxes1[:, :, :, :, 3] + 1e-9))
    atan2 = L.atan(boxes2[:, :, :, :, 2] / (boxes2[:, :, :, :, 3] + 1e-9))
    v = 4.0 * L.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou


def bbox_iou(boxes1, boxes2):
    '''
    预测框          boxes1 (?, grid_h, grid_w, 3,   1, 4)，神经网络的输出(tx, ty, tw, th)经过了后处理求得的(bx, by, bw, bh)
    图片中所有的gt  boxes2 (?,      1,      1, 1, 70, 4)
    '''
    boxes1_area = boxes1[:, :, :, :, :, 2] * boxes1[:, :, :, :, :, 3]  # 所有格子的3个预测框的面积
    boxes2_area = boxes2[:, :, :, :, :, 2] * boxes2[:, :, :, :, :, 3]  # 所有ground truth的面积

    # (x, y, w, h)变成(x0, y0, x1, y1)
    boxes1 = L.concat([boxes1[:, :, :, :, :, :2] - boxes1[:, :, :, :, :, 2:] * 0.5,
                       boxes1[:, :, :, :, :, :2] + boxes1[:, :, :, :, :, 2:] * 0.5], axis=-1)
    boxes2 = L.concat([boxes2[:, :, :, :, :, :2] - boxes2[:, :, :, :, :, 2:] * 0.5,
                       boxes2[:, :, :, :, :, :2] + boxes2[:, :, :, :, :, 2:] * 0.5], axis=-1)

    # 所有格子的3个预测框 分别 和  70个ground truth  计算iou。 所以left_up和right_down的shape = (?, grid_h, grid_w, 3, 70, 2)
    expand_boxes1 = L.expand(boxes1, [1, 1, 1, 1, L.shape(boxes2)[4], 1])  # 不同于pytorch和tf，boxes1和boxes2都要扩展为相同shape
    expand_boxes2 = L.expand(boxes2, [1, L.shape(boxes1)[1], L.shape(boxes1)[2], L.shape(boxes1)[3], 1,
                                      1])  # 不同于pytorch和tf，boxes1和boxes2都要扩展为相同shape
    left_up = L.elementwise_max(expand_boxes1[:, :, :, :, :, :2], expand_boxes2[:, :, :, :, :, :2])  # 相交矩形的左上角坐标
    right_down = L.elementwise_min(expand_boxes1[:, :, :, :, :, 2:], expand_boxes2[:, :, :, :, :, 2:])  # 相交矩形的右下角坐标

    inter_section = L.relu(right_down - left_up)  # 相交矩形的w和h，是负数时取0  (?, grid_h, grid_w, 3, 70, 2)
    inter_area = inter_section[:, :, :, :, :, 0] * inter_section[:, :, :, :, :, 1]  # 相交矩形的面积   (?, grid_h, grid_w, 3, 70)
    expand_boxes1_area = L.expand(boxes1_area, [1, 1, 1, 1, L.shape(boxes2)[4]])
    expand_boxes2_area = L.expand(boxes2_area, [1, L.shape(expand_boxes1_area)[1], L.shape(expand_boxes1_area)[2],
                                                L.shape(expand_boxes1_area)[3], 1])
    union_area = expand_boxes1_area + expand_boxes2_area - inter_area  # union_area                (?, grid_h, grid_w, 3, 70)
    iou = inter_area / (union_area + 1e-9)  # iou                       (?, grid_h, grid_w, 3, 70)

    return iou



class MyLoss(object):
    """
    Combined loss for YOLOv3 network

    Args:
        ignore_thresh (float): threshold to ignore confidence loss
        label_smooth (bool): whether to use label smoothing
        use_fine_grained_loss (bool): whether use fine grained YOLOv3 loss
                                      instead of fluid.layers.yolov3_loss
    """

    def __init__(self,
                 ignore_thresh=0.7,
                 label_smooth=True,
                 use_fine_grained_loss=False,
                 iou_loss=None,
                 iou_aware_loss=None,
                 downsample=[32, 16, 8],
                 scale_x_y=1.,
                 match_score=False):
        self._ignore_thresh = ignore_thresh
        self._label_smooth = label_smooth
        self._use_fine_grained_loss = use_fine_grained_loss
        self._iou_loss = iou_loss
        self._iou_aware_loss = iou_aware_loss
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.match_score = match_score

    def __call__(self, outputs, gt_box, gt_label, gt_score, targets, anchors,
                 anchor_masks, mask_anchors, num_classes):
        return self._get_fine_grained_loss(
            outputs, targets, gt_box, num_classes,
            mask_anchors, self._ignore_thresh)

    def _get_fine_grained_loss(self,
                               outputs,
                               targets,
                               gt_box,
                               num_classes,
                               mask_anchors,
                               ignore_thresh,
                               eps=1.e-10):
        """
        Calculate fine grained YOLOv3 loss

        Args:
            outputs ([Variables]): List of Variables, output of backbone stages
            targets ([Variables]): List of Variables, The targets for yolo
                                   loss calculatation.
            gt_box (Variable): The ground-truth boudding boxes.
            num_classes (int): class num of dataset
            mask_anchors ([[float]]): list of anchors in each output layer
            ignore_thresh (float): prediction bbox overlap any gt_box greater
                                   than ignore_thresh, objectness loss will
                                   be ignored.

        Returns:
            Type: dict
                xy_loss (Variable): YOLOv3 (x, y) coordinates loss
                wh_loss (Variable): YOLOv3 (w, h) coordinates loss
                obj_loss (Variable): YOLOv3 objectness score loss
                cls_loss (Variable): YOLOv3 classification loss

        """

        assert len(outputs) == len(targets), \
            "YOLOv3 output layer number not equal target number"

        batch_size = gt_box.shape[0]
        loss_boxes, loss_objs, loss_clss = [], [], []
        if self._iou_loss is not None:
            loss_ious = []
        if self._iou_aware_loss is not None:
            loss_iou_awares = []
        for i, (output, target,
                anchors) in enumerate(zip(outputs, targets, mask_anchors)):
            downsample = self.downsample[i]
            an_num = len(anchors) // 2
            scale_x_y = self.scale_x_y if not isinstance(
                self.scale_x_y, Sequence) else self.scale_x_y[i]

            target = L.transpose(target, perm=[0, 3, 4, 1, 2])   # [N, 3, 86, 13, 13] -> [N, 13, 13, 3, 86]
            output = L.transpose(output, perm=[0, 2, 3, 1])      # [N, 255, 13, 13] -> [N, 13, 13, 255]
            anchors = np.array(anchors).astype(np.float32)
            anchors = np.reshape(anchors, (-1, 2))
            pred_bbox = self._decode(output, anchors, downsample, num_classes, scale_x_y, eps)

            pred_shape = pred_bbox.shape
            batch_size = pred_shape[0]
            output_size = pred_shape[1]

            pred_xywh = pred_bbox[:, :, :, :, 0:4]
            pred_conf = pred_bbox[:, :, :, :, 4:5]
            pred_prob = pred_bbox[:, :, :, :, 5:]

            label_xywh = target[:, :, :, :, 0:4]
            tobj       = target[:, :, :, :, 4:5]
            tscale     = target[:, :, :, :, 5:6]
            label_prob = target[:, :, :, :, 6:]

            ciou = bbox_ciou(pred_xywh, label_xywh)  # (8, 13, 13, 3)
            ciou = L.reshape(ciou, (batch_size, output_size, output_size, 3, 1))  # (8, 13, 13, 3, 1)

            # 每个预测框xxxiou_loss的权重 tscale = 2 - (ground truth的面积/图片面积)
            ciou_loss = tobj * tscale * (1 - ciou)  # 1. tobj作为mask，有物体才计算xxxiou_loss

            # 2. respond_bbox作为mask，有物体才计算类别loss
            prob_pos_loss = label_prob * (0 - L.log(pred_prob + 1e-9))  # 二值交叉熵，tf中也是加了极小的常数防止nan
            prob_neg_loss = (1 - label_prob) * (0 - L.log(1 - pred_prob + 1e-9))  # 二值交叉熵，tf中也是加了极小的常数防止nan
            prob_mask = L.expand(tobj, [1, 1, 1, 1, num_classes])
            prob_loss = prob_mask * (prob_pos_loss + prob_neg_loss)

            # 3. xxxiou_loss和类别loss比较简单。重要的是conf_loss，是一个二值交叉熵损失
            # 分两步：第一步是确定 grid_h * grid_w * 3 个预测框 哪些作为反例；第二步是计算二值交叉熵损失。
            expand_pred_xywh = L.reshape(pred_xywh, (batch_size, output_size, output_size, 3, 1, 4))  # 扩展为(?, grid_h, grid_w, 3,   1, 4)
            # gt_box为cx_cy_w_h格式
            expand_bboxes = L.reshape(gt_box, (batch_size, 1, 1, 1, L.shape(gt_box)[1], 4))  # 扩展为(?,      1,      1, 1, 70, 4)
            iou = bbox_iou(expand_pred_xywh, expand_bboxes)  # 所有格子的3个预测框 分别 和  70个ground truth  计算iou。   (?, grid_h, grid_w, 3, 70)
            max_iou, max_iou_indices = L.topk(iou, k=1)  # 与70个ground truth的iou中，保留最大那个iou。  (?, grid_h, grid_w, 3, 1)

            # respond_bgd代表  这个分支输出的 grid_h * grid_w * 3 个预测框是否是 反例（背景）
            # label有物体，respond_bgd是0。 没物体的话：如果和某个gt(共70个)的iou超过iou_loss_thresh，respond_bgd是0；如果和所有gt(最多70个)的iou都小于iou_loss_thresh，respond_bgd是1。
            # respond_bgd是0代表有物体，不是反例（或者是忽略框）；  权重respond_bgd是1代表没有物体，是反例。
            # 有趣的是，模型训练时由于不断更新，对于同一张图片，两次预测的 grid_h * grid_w * 3 个预测框（对于这个分支输出）  是不同的。用的是这些预测框来与gt计算iou来确定哪些预测框是反例。
            # 而不是用固定大小（不固定位置）的先验框。
            respond_bgd = (1.0 - tobj) * L.cast(max_iou < self._ignore_thresh, 'float32')

            # 二值交叉熵损失
            pos_loss = tobj * (0 - L.log(pred_conf + 1e-9))
            neg_loss = respond_bgd * (0 - L.log(1 - pred_conf + 1e-9))

            conf_loss = pos_loss + neg_loss
            # 回顾respond_bgd，某个预测框和某个gt的iou超过iou_loss_thresh，不被当作是反例。在参与“预测的置信位 和 真实置信位 的 二值交叉熵”时，这个框也可能不是正例(label里没标这个框是1的话)。这个框有可能不参与置信度loss的计算。
            # 这种框一般是gt框附近的框，或者是gt框所在格子的另外两个框。它既不是正例也不是反例不参与置信度loss的计算。（论文里称之为ignore）

            ciou_loss = L.reduce_sum(ciou_loss) / batch_size
            conf_loss = L.reduce_sum(conf_loss) / batch_size
            prob_loss = L.reduce_sum(prob_loss) / batch_size

            loss_boxes.append(ciou_loss)
            loss_objs.append(conf_loss)
            loss_clss.append(prob_loss)

        losses_all = {
            "loss_box": fluid.layers.sum(loss_boxes),
            "loss_obj": fluid.layers.sum(loss_objs),
            "loss_cls": fluid.layers.sum(loss_clss),
        }
        # if self._iou_loss is not None:
        #     losses_all["loss_iou"] = fluid.layers.sum(loss_ious)
        if self._iou_aware_loss is not None:
            losses_all["loss_iou_aware"] = fluid.layers.sum(loss_iou_awares)
        return losses_all

    def _decode(self, conv_output, anchors, stride, num_class, scale_x_y, eps):
        conv_shape       = conv_output.shape
        batch_size       = conv_shape[0]
        n_grid           = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = L.reshape(conv_output, (batch_size, n_grid, n_grid, anchor_per_scale, 5 + num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        rows = L.range(0, n_grid, 1, 'float32')
        cols = L.range(0, n_grid, 1, 'float32')
        rows = L.expand(L.reshape(rows, (1, -1, 1)), [n_grid, 1, 1])
        cols = L.expand(L.reshape(cols, (-1, 1, 1)), [1, n_grid, 1])
        offset = L.concat([rows, cols], axis=-1)
        offset = L.reshape(offset, (1, n_grid, n_grid, 1, 2))
        offset = L.expand(offset, [batch_size, 1, 1, anchor_per_scale, 1])

        if (abs(scale_x_y - 1.0) < eps):
            pred_xy = L.sigmoid(conv_raw_dxdy)
            pred_xy = (pred_xy + offset) * stride
        else:
            # Grid Sensitive
            pred_xy = scale_x_y * L.sigmoid(conv_raw_dxdy) - 0.5 * (scale_x_y - 1.0)
            pred_xy = (pred_xy + offset) * stride
        anchor_t = fluid.layers.assign(np.copy(anchors).astype(np.float32))
        pred_wh = (L.exp(conv_raw_dwdh) * anchor_t)
        pred_xywh = L.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = L.sigmoid(conv_raw_conf)
        pred_prob = L.sigmoid(conv_raw_prob)

        return L.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def _split_output(self, output, an_num, num_classes):
        """
        Split output feature map to x, y, w, h, objectness, classification
        along channel dimension
        """
        batch_size = output.shape[0]
        output_size = output.shape[2]
        output = L.reshape(output, (batch_size, an_num, 5 + num_classes, output_size, output_size))
        x = output[:, :, 0, :, :]
        y = output[:, :, 1, :, :]
        w = output[:, :, 2, :, :]
        h = output[:, :, 3, :, :]
        obj = output[:, :, 4, :, :]
        cls = output[:, :, 5:, :, :]
        cls = L.transpose(cls, perm=[0, 1, 3, 4, 2])
        return (x, y, w, h, obj, cls)

    def _split_target(self, target):
        """
        split target to x, y, w, h, objectness, classification
        along dimension 2

        target is in shape [N, an_num, 6 + class_num, H, W]
        """
        tx = target[:, :, 0, :, :]
        ty = target[:, :, 1, :, :]
        tw = target[:, :, 2, :, :]
        th = target[:, :, 3, :, :]

        tscale = target[:, :, 4, :, :]
        tobj = target[:, :, 5, :, :]

        tcls = target[:, :, 6:, :, :]
        tcls = L.transpose(tcls, perm=[0, 1, 3, 4, 2])
        tcls.stop_gradient = True

        return (tx, ty, tw, th, tscale, tobj, tcls)

    def _calc_obj_loss(self, output, obj, tobj, gt_box, batch_size, anchors,
                       num_classes, downsample, ignore_thresh, scale_x_y):
        # A prediction bbox overlap any gt_bbox over ignore_thresh,
        # objectness loss will be ignored, process as follows:

        # 1. get pred bbox, which is same with YOLOv3 infer mode, use yolo_box here
        # NOTE: img_size is set as 1.0 to get noramlized pred bbox
        bbox, prob = fluid.layers.yolo_box(
            x=output,
            img_size=fluid.layers.ones(
                shape=[batch_size, 2], dtype="int32"),
            anchors=anchors,
            class_num=num_classes,
            conf_thresh=0.,
            downsample_ratio=downsample,
            clip_bbox=False,
            scale_x_y=scale_x_y)

        # 2. split pred bbox and gt bbox by sample, calculate IoU between pred bbox
        #    and gt bbox in each sample
        if batch_size > 1:
            preds = fluid.layers.split(bbox, batch_size, dim=0)
            gts = fluid.layers.split(gt_box, batch_size, dim=0)
        else:
            preds = [bbox]
            gts = [gt_box]
            probs = [prob]
        ious = []
        for pred, gt in zip(preds, gts):

            def box_xywh2xyxy(box):
                x = box[:, 0]
                y = box[:, 1]
                w = box[:, 2]
                h = box[:, 3]
                return fluid.layers.stack(
                    [
                        x - w / 2.,
                        y - h / 2.,
                        x + w / 2.,
                        y + h / 2.,
                    ], axis=1)

            pred = fluid.layers.squeeze(pred, axes=[0])
            gt = box_xywh2xyxy(fluid.layers.squeeze(gt, axes=[0]))
            ious.append(fluid.layers.iou_similarity(pred, gt))

        iou = fluid.layers.stack(ious, axis=0)
        # 3. Get iou_mask by IoU between gt bbox and prediction bbox,
        #    Get obj_mask by tobj(holds gt_score), calculate objectness loss

        max_iou = fluid.layers.reduce_max(iou, dim=-1)
        iou_mask = fluid.layers.cast(max_iou <= ignore_thresh, dtype="float32")
        if self.match_score:
            max_prob = fluid.layers.reduce_max(prob, dim=-1)
            iou_mask = iou_mask * fluid.layers.cast(
                max_prob <= 0.25, dtype="float32")
        output_shape = fluid.layers.shape(output)
        an_num = len(anchors) // 2
        iou_mask = fluid.layers.reshape(iou_mask, (-1, an_num, output_shape[2],
                                                   output_shape[3]))
        iou_mask.stop_gradient = True

        # NOTE: tobj holds gt_score, obj_mask holds object existence mask
        obj_mask = fluid.layers.cast(tobj > 0., dtype="float32")
        obj_mask.stop_gradient = True

        # For positive objectness grids, objectness loss should be calculated
        # For negative objectness grids, objectness loss is calculated only iou_mask == 1.0
        loss_obj = fluid.layers.sigmoid_cross_entropy_with_logits(obj, obj_mask)
        loss_obj_pos = fluid.layers.reduce_sum(loss_obj * tobj, dim=[1, 2, 3])
        loss_obj_neg = fluid.layers.reduce_sum(
            loss_obj * (1.0 - obj_mask) * iou_mask, dim=[1, 2, 3])

        return loss_obj_pos, loss_obj_neg




