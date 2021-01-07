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



# 相交矩形的面积
def intersect(box_a, box_b):
    """计算两组矩形两两之间相交区域的面积
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
      (tensor) intersection area, Shape: [A, B].
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
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """计算两组矩形两两之间的iou
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
        ious: (tensor) Shape: [A, B]
    """
    inter = intersect(box_a, box_b)

    A = box_a.shape[0]
    B = box_b.shape[0]

    area_a = (box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])
    area_a = L.reshape(area_a, (A, 1))
    area_a = L.expand(area_a, [1, B])  # [A, B]

    area_b = (box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])
    area_b = L.reshape(area_b, (1, B))
    area_b = L.expand(area_b, [A, 1])  # [A, B]


    union = area_a + area_b - inter
    return inter / union  # [A, B]



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
    compensate_iou = L.transpose(L.expand(L.reshape(compensate_iou, (1, -1)), [n_samples, 1]), [1, 0])   # shape: [n_samples, n_samples]

    # IoU decay
    # 非同类的iou置为0，同类的iou保留。
    decay_iou = iou_matrix * label_matrix   # shape: [n_samples, n_samples]

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = L.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = L.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient = L.reduce_sum(decay_matrix / compensate_matrix, [0, ])
    elif kernel == 'linear':
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



