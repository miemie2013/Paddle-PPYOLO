#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import paddle
import paddle.fluid.layers as L
import math



def _iou(box_a, box_b):
    """计算两组矩形两两之间的iou
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
      (tensor) iou, Shape: [A, B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]

    box_a_rb = L.reshape(box_a[:, 2:], (A, 1, 2))
    box_a_rb = L.expand(box_a_rb, [1, B, 1])
    box_b_rb = L.reshape(box_b[:, 2:], (1, B, 2))
    box_b_rb = L.expand(box_b_rb, [A, 1, 1])
    max_xy = L.elementwise_min(box_a_rb, box_b_rb)

    box_a_lu = L.reshape(box_a[:, :2], (A, 1, 2))
    box_a_lu = L.expand(box_a_lu, [1, B, 1])
    box_b_lu = L.reshape(box_b[:, :2], (1, B, 2))
    box_b_lu = L.expand(box_b_lu, [A, 1, 1])
    min_xy = L.elementwise_max(box_a_lu, box_b_lu)

    inter = L.relu(max_xy - min_xy)
    inter = inter[:, :, 0] * inter[:, :, 1]

    box_a_w = box_a[:, 2]-box_a[:, 0]
    box_a_h = box_a[:, 3]-box_a[:, 1]
    area_a = box_a_h * box_a_w
    area_a = L.reshape(area_a, (A, 1))
    area_a = L.expand(area_a, [1, B])  # [A, B]

    box_b_w = box_b[:, 2]-box_b[:, 0]
    box_b_h = box_b[:, 3]-box_b[:, 1]
    area_b = box_b_h * box_b_w
    area_b = L.reshape(area_b, (1, B))
    area_b = L.expand(area_b, [A, 1])  # [A, B]

    union = area_a + area_b - inter
    return inter / union  # [A, B]


def _iou_hw(box_a, box_b, eps=1e-9):
    """计算两组矩形两两之间的iou以及长宽比信息
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
      (tensor) iou, Shape: [A, B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]

    box_a_rb = L.reshape(box_a[:, 2:], (A, 1, 2))
    box_a_rb = L.expand(box_a_rb, [1, B, 1])
    box_b_rb = L.reshape(box_b[:, 2:], (1, B, 2))
    box_b_rb = L.expand(box_b_rb, [A, 1, 1])
    max_xy = L.elementwise_min(box_a_rb, box_b_rb)

    box_a_lu = L.reshape(box_a[:, :2], (A, 1, 2))
    box_a_lu = L.expand(box_a_lu, [1, B, 1])
    box_b_lu = L.reshape(box_b[:, :2], (1, B, 2))
    box_b_lu = L.expand(box_b_lu, [A, 1, 1])
    min_xy = L.elementwise_max(box_a_lu, box_b_lu)

    inter = L.relu(max_xy - min_xy)
    inter = inter[:, :, 0] * inter[:, :, 1]

    box_a_w = box_a[:, 2]-box_a[:, 0]
    box_a_h = box_a[:, 3]-box_a[:, 1]
    area_a = box_a_h * box_a_w
    area_a = L.reshape(area_a, (A, 1))
    area_a = L.expand(area_a, [1, B])  # [A, B]

    box_b_w = box_b[:, 2]-box_b[:, 0]
    box_b_h = box_b[:, 3]-box_b[:, 1]
    area_b = box_b_h * box_b_w
    area_b = L.reshape(area_b, (1, B))
    area_b = L.expand(area_b, [A, 1])  # [A, B]

    union = area_a + area_b - inter
    iou = inter / union  # [A, B]  iou取值0~1之间，iou越大越应该抑制

    # 长宽比信息
    atan1 = L.atan(box_a_h / (box_a_w + eps))
    atan2 = L.atan(box_b_h / (box_b_w + eps))
    atan1 = L.reshape(atan1, (A, 1))
    atan1 = L.expand(atan1, [1, B])  # [A, B]
    atan2 = L.reshape(atan2, (1, B))
    atan2 = L.expand(atan2, [A, 1])  # [A, B]
    v = 4.0 * L.pow(atan1 - atan2, 2) / (math.pi ** 2)  # [A, B]  v取值0~1之间，v越小越应该抑制

    factor = 0.4
    overlap = L.pow(iou, (1 - factor)) * L.pow(1.0 - v, factor)

    return overlap


def jaccard(box_a, box_b, type='iou'):
    """计算两组矩形两两之间的重叠
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
        overlap: (tensor) Shape: [A, B]
    """
    if type == 'iou':
        overlap = _iou(box_a, box_b)
    elif type == 'iou_hw':
        overlap = _iou_hw(box_a, box_b)
    return overlap



def _matrix_nms(bboxes, cate_labels, cate_scores, kernel='gaussian', sigma=2.0):
    """Matrix NMS for multi-class bboxes.
    Args:
        bboxes (Tensor): shape (n, 4)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gaussian'
        sigma (float): std in gaussian method
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []

    # 计算一个n×n的IOU矩阵，两组矩形两两之间的IOU
    iou_matrix = jaccard(bboxes, bboxes)   # shape: [n_samples, n_samples]
    iou_matrix = paddle.triu(iou_matrix, diagonal=1)   # 只取上三角部分

    # label_specific matrix.
    cate_labels_x = L.expand(L.reshape(cate_labels, (1, -1)), [n_samples, 1])   # shape: [n_samples, n_samples]
    # 第i行第j列表示的是第i个预测框和第j个预测框的类别id是否相同。我们抑制的是同类的预测框。
    d = cate_labels_x - L.transpose(cate_labels_x, [1, 0])
    d = L.pow(d, 2)   # 同类处为0，非同类处>0。 tf中用 == 0比较无效，所以用 < 1
    label_matrix = paddle.triu(L.cast(d < 1, 'float32'), diagonal=1)   # shape: [n_samples, n_samples]

    # IoU compensation
    # 非同类的iou置为0，同类的iou保留。逐列取最大iou
    compensate_iou = L.reduce_max(iou_matrix * label_matrix, [0, ])   # shape: [n_samples, ]
    # compensate_iou第0行里的值a0（重复了n_samples次）表示第0个物体与 比它分高 的 同类物体的最高iou为a0，
    # compensate_iou第1行里的值a1（重复了n_samples次）表示第1个物体与 比它分高 的 同类物体的最高iou为a1，...
    # compensate_iou里每一列里的值依次代表第0个物体、第1个物体、...、第n_samples-1个物体与 比它自己分高 的 同类物体的最高iou。
    compensate_iou = L.transpose(L.expand(L.reshape(compensate_iou, (1, -1)), [n_samples, 1]), [1, 0])   # shape: [n_samples, n_samples]

    # IoU decay
    # 非同类的iou置为0，同类的iou保留。
    # decay_iou第i行第j列表示的是第i个预测框和第j个预测框的iou，如果不是同类，该iou置0。且只取上三角部分。
    decay_iou = iou_matrix * label_matrix   # shape: [n_samples, n_samples]

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = L.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = L.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient = L.reduce_sum(decay_matrix / compensate_matrix, [0, ])
    elif kernel == 'linear':
        # 看第j列。（1_test_matrixnms.py里的例子，看第2列）
        # decay_iou     里第2列里的值为[0.9389, 0.9979, 0,      0]。第2个物体与比它分高的2个同类物体的iou是0.9389, 0.9979。
        # compensate_iou里第2列里的值为[0,      0.9409, 0.9979, 0]。比第2个物体分高的2个同类物体 与 比它们自己分高 的 同类物体的最高iou 是0,      0.9409。
        # decay_matrix  里第2列里的值为[0.0610, 0.0348, 485.28, 1]。取该列的最小值为0.0348（抑制掉第2个物体的是第1个物体）。其实后面2个值不用看，因为它们总是>=1。
        # 总结：decay_matrix里第j列里的第i个值若为最小值，则抑制掉第j个物体的是第i个物体。
        # 而且，表现为decay_iou尽可能大，decay_matrix才会尽可能小。
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient = L.reduce_min(decay_matrix, [0, ])
    else:
        raise NotImplementedError

    # 更新分数
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update




def matrix_nms(bboxes,
               scores,
               score_threshold,
               post_threshold,
               nms_top_k,
               keep_top_k,
               use_gaussian=False,
               gaussian_sigma=2.):
    scores = L.transpose(scores, [1, 0])
    inds = L.where(scores > score_threshold)
    if len(inds) == 0:
        return L.zeros((0, 6), 'float32') - 1.0

    cate_scores = L.gather_nd(scores, inds)
    cate_labels = inds[:, 1]
    bboxes = L.gather(bboxes, inds[:, 0])

    # sort and keep top nms_top_k
    _, sort_inds = L.argsort(cate_scores, descending=True)
    if nms_top_k > 0 and len(sort_inds) > nms_top_k:
        sort_inds = sort_inds[:nms_top_k]
    bboxes = L.gather(bboxes, sort_inds)
    cate_scores = L.gather(cate_scores, sort_inds)
    cate_labels = L.gather(cate_labels, sort_inds)

    # Matrix NMS
    kernel = 'gaussian' if use_gaussian else 'linear'
    cate_scores = _matrix_nms(bboxes, cate_labels, cate_scores, kernel=kernel, sigma=gaussian_sigma)

    # filter.
    keep = L.where(cate_scores >= post_threshold)
    if len(keep) == 0:
        return L.zeros((0, 6), 'float32') - 1.0
    bboxes = L.gather(bboxes, keep)
    cate_scores = L.gather(cate_scores, keep)
    cate_labels = L.gather(cate_labels, keep)

    # sort and keep keep_top_k
    _, sort_inds = L.argsort(cate_scores, descending=True)
    if len(sort_inds) > keep_top_k:
        sort_inds = sort_inds[:keep_top_k]
    bboxes = L.gather(bboxes, sort_inds)
    cate_scores = L.gather(cate_scores, sort_inds)
    cate_labels = L.gather(cate_labels, sort_inds)

    cate_scores = L.unsqueeze(cate_scores, 1)
    cate_labels = L.unsqueeze(cate_labels, 1)
    cate_labels = L.cast(cate_labels, 'float32')
    pred = L.concat([cate_labels, cate_scores, bboxes], 1)

    return pred



def no_nms(bboxes,
           scores,
           score_threshold,
           keep_top_k):
    scores = L.transpose(scores, [1, 0])
    inds = L.where(scores > score_threshold)
    if len(inds) == 0:
        return L.zeros((0, 6), 'float32') - 1.0

    cate_scores = L.gather_nd(scores, inds)
    cate_labels = inds[:, 1]
    bboxes = L.gather(bboxes, inds[:, 0])

    # sort and keep top keep_top_k
    _, sort_inds = L.argsort(cate_scores, descending=True)
    if keep_top_k > 0 and len(sort_inds) > keep_top_k:
        sort_inds = sort_inds[:keep_top_k]
    bboxes = L.gather(bboxes, sort_inds)
    cate_scores = L.gather(cate_scores, sort_inds)
    cate_labels = L.gather(cate_labels, sort_inds)

    cate_scores = L.unsqueeze(cate_scores, 1)
    cate_labels = L.unsqueeze(cate_labels, 1)
    cate_labels = L.cast(cate_labels, 'float32')
    pred = L.concat([cate_labels, cate_scores, bboxes], 1)

    return pred



