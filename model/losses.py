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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence



class YOLOv3Loss(object):
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
                 focalloss_on_obj=False,
                 focalloss_alpha=0.25,
                 focalloss_gamma=2.0,
                 match_score=False):
        self._ignore_thresh = ignore_thresh
        self._label_smooth = label_smooth
        self._use_fine_grained_loss = use_fine_grained_loss
        self._iou_loss = iou_loss
        self._iou_aware_loss = iou_aware_loss
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.match_score = match_score
        self.focalloss_on_obj = focalloss_on_obj
        self.focalloss_alpha = focalloss_alpha
        self.focalloss_gamma = focalloss_gamma

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
        loss_xys, loss_whs, loss_objs, loss_clss = [], [], [], []
        if self._iou_loss is not None:
            loss_ious = []
        if self._iou_aware_loss is not None:
            loss_iou_awares = []
        for i, (output, target,
                anchors) in enumerate(zip(outputs, targets, mask_anchors)):
            downsample = self.downsample[i]
            an_num = len(anchors) // 2
            if self._iou_aware_loss is not None:
                ioup, output = self._split_ioup(output, an_num, num_classes)
            x, y, w, h, obj, cls = self._split_output(output, an_num,
                                                      num_classes)
            tx, ty, tw, th, tscale, tobj, tcls = self._split_target(target)

            tscale_tobj = tscale * tobj

            scale_x_y = self.scale_x_y if not isinstance(
                self.scale_x_y, Sequence) else self.scale_x_y[i]

            if (abs(scale_x_y - 1.0) < eps):
                loss_x = fluid.layers.sigmoid_cross_entropy_with_logits(x, tx) * tscale_tobj
                loss_x = fluid.layers.reduce_sum(loss_x, dim=[1, 2, 3])
                loss_y = fluid.layers.sigmoid_cross_entropy_with_logits(y, ty) * tscale_tobj
                loss_y = fluid.layers.reduce_sum(loss_y, dim=[1, 2, 3])
            else:
                dx = scale_x_y * fluid.layers.sigmoid(x) - 0.5 * (scale_x_y - 1.0)
                dy = scale_x_y * fluid.layers.sigmoid(y) - 0.5 * (scale_x_y - 1.0)
                loss_x = fluid.layers.abs(dx - tx) * tscale_tobj
                loss_x = fluid.layers.reduce_sum(loss_x, dim=[1, 2, 3])
                loss_y = fluid.layers.abs(dy - ty) * tscale_tobj
                loss_y = fluid.layers.reduce_sum(loss_y, dim=[1, 2, 3])

            # NOTE: we refined loss function of (w, h) as L1Loss
            loss_w = fluid.layers.abs(w - tw) * tscale_tobj
            loss_w = fluid.layers.reduce_sum(loss_w, dim=[1, 2, 3])
            loss_h = fluid.layers.abs(h - th) * tscale_tobj
            loss_h = fluid.layers.reduce_sum(loss_h, dim=[1, 2, 3])
            if self._iou_loss is not None:
                loss_iou = self._iou_loss(x, y, w, h, tx, ty, tw, th, anchors,
                                          downsample, batch_size,
                                          scale_x_y)
                loss_iou = loss_iou * tscale_tobj
                loss_iou = fluid.layers.reduce_sum(loss_iou, dim=[1, 2, 3])
                loss_ious.append(fluid.layers.reduce_mean(loss_iou))

            if self._iou_aware_loss is not None:
                loss_iou_aware = self._iou_aware_loss(
                    ioup, x, y, w, h, tx, ty, tw, th, anchors, downsample,
                    batch_size, scale_x_y)
                loss_iou_aware = loss_iou_aware * tobj
                loss_iou_aware = fluid.layers.reduce_sum(
                    loss_iou_aware, dim=[1, 2, 3])
                loss_iou_awares.append(fluid.layers.reduce_mean(loss_iou_aware))

            loss_obj_pos, loss_obj_neg = self._calc_obj_loss(
                output, obj, tobj, gt_box, batch_size, anchors,
                num_classes, downsample, self._ignore_thresh, scale_x_y)

            loss_cls = fluid.layers.sigmoid_cross_entropy_with_logits(cls, tcls)
            loss_cls = fluid.layers.elementwise_mul(loss_cls, tobj, axis=0)
            loss_cls = fluid.layers.reduce_sum(loss_cls, dim=[1, 2, 3, 4])

            loss_xys.append(fluid.layers.reduce_mean(loss_x + loss_y))
            loss_whs.append(fluid.layers.reduce_mean(loss_w + loss_h))
            loss_objs.append(
                fluid.layers.reduce_mean(loss_obj_pos + loss_obj_neg))
            loss_clss.append(fluid.layers.reduce_mean(loss_cls))

        losses_all = {
            "loss_xy": fluid.layers.sum(loss_xys),
            "loss_wh": fluid.layers.sum(loss_whs),
            "loss_obj": fluid.layers.sum(loss_objs),
            "loss_cls": fluid.layers.sum(loss_clss),
        }
        if self._iou_loss is not None:
            losses_all["loss_iou"] = fluid.layers.sum(loss_ious)
        if self._iou_aware_loss is not None:
            losses_all["loss_iou_aware"] = fluid.layers.sum(loss_iou_awares)
        return losses_all

    def _split_ioup(self, output, an_num, num_classes):
        """
        Split output feature map to output, predicted iou
        along channel dimension
        """
        ioup = output[:, :an_num, :, :]
        ioup = L.sigmoid(ioup)

        oriout = output[:, an_num:, :, :]
        return (ioup, oriout)

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
            # bbox:    [N, 3*n_grid*n_grid, 4]
            # gt_box:  [N, 50, 4]
            preds = fluid.layers.split(bbox, batch_size, dim=0)   # 里面每个元素形状是[1, 3*n_grid*n_grid, 4]
            gts = fluid.layers.split(gt_box, batch_size, dim=0)   # 里面每个元素形状是[1, 50, 4]
        else:
            preds = [bbox]
            gts = [gt_box]
            probs = [prob]
        ious = []
        for pred, gt in zip(preds, gts):
            # pred:   [1, 3*n_grid*n_grid, 4]
            # gt:     [1, 50, 4]

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

            pred = fluid.layers.squeeze(pred, axes=[0])   # [3*n_grid*n_grid, 4]
            gt = box_xywh2xyxy(fluid.layers.squeeze(gt, axes=[0]))   # [50, 4]  且转换成x0y0x1y1格式
            ious.append(fluid.layers.iou_similarity(pred, gt))   # [3*n_grid*n_grid, 50]   两组矩形两两之间的iou

        iou = fluid.layers.stack(ious, axis=0)   # [N, 3*n_grid*n_grid, 50]   两组矩形两两之间的iou
        # 3. Get iou_mask by IoU between gt bbox and prediction bbox,
        #    Get obj_mask by tobj(holds gt_score), calculate objectness loss

        max_iou = fluid.layers.reduce_max(iou, dim=-1)   # [N, 3*n_grid*n_grid]   预测框和本图片所有gt的最高iou
        iou_mask = fluid.layers.cast(max_iou <= ignore_thresh, dtype="float32")   # [N, 3*n_grid*n_grid]   候选负样本处为1
        if self.match_score:
            max_prob = fluid.layers.reduce_max(prob, dim=-1)
            iou_mask = iou_mask * fluid.layers.cast(
                max_prob <= 0.25, dtype="float32")
        output_shape = fluid.layers.shape(output)
        an_num = len(anchors) // 2
        iou_mask = fluid.layers.reshape(iou_mask, (-1, an_num, output_shape[2],
                                                   output_shape[3]))   # [N, 3, n_grid, n_grid]   候选负样本处为1
        iou_mask.stop_gradient = True

        # NOTE: tobj holds gt_score, obj_mask holds object existence mask
        obj_mask = fluid.layers.cast(tobj > 0., dtype="float32")   # [N, 3, n_grid, n_grid]  正样本处为1
        obj_mask.stop_gradient = True

        noobj_mask = (1.0 - obj_mask) * iou_mask   # [N, 3, n_grid, n_grid]  负样本处为1
        noobj_mask.stop_gradient = True

        # For positive objectness grids, objectness loss should be calculated
        # For negative objectness grids, objectness loss is calculated only iou_mask == 1.0
        pred_conf = L.sigmoid(obj)
        if self.focalloss_on_obj:
            alpha = self.focalloss_alpha
            gamma = self.focalloss_gamma
            pos_loss = tobj * (0 - L.log(pred_conf + 1e-9)) * L.pow(1 - pred_conf, gamma) * alpha
            neg_loss = noobj_mask * (0 - L.log(1 - pred_conf + 1e-9)) * L.pow(pred_conf, gamma) * (1 - alpha)
        else:
            pos_loss = tobj * (0 - L.log(pred_conf + 1e-9))
            neg_loss = noobj_mask * (0 - L.log(1 - pred_conf + 1e-9))
        pos_loss = fluid.layers.reduce_sum(pos_loss, dim=[1, 2, 3])
        neg_loss = fluid.layers.reduce_sum(neg_loss, dim=[1, 2, 3])

        return pos_loss, neg_loss




