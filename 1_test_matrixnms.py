#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import os
import paddle
import paddle.fluid.layers as L
import numpy as np
from model.matrix_nms import _matrix_nms

use_gpu = True


print(paddle.__version__)
paddle.disable_static()  # 开启动态图
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()



# 假设有4个预测框，前3个是同一类别的。很明显bboxes[1]、bboxes[2]会被抑制。
bboxes = np.zeros((4, 4))
bboxes[0] = np.array([0, 0, 100, 100])
bboxes[1] = np.array([3, 3, 100, 100])
bboxes[2] = np.array([3.1, 3.1, 100, 100])
bboxes[3] = np.array([50, 50, 100, 100])
bboxes = paddle.to_tensor(bboxes, place=place)
bboxes = L.cast(bboxes, 'float32')

cate_labels = np.zeros((4, ))
cate_labels[0] = 19
cate_labels[1] = 19
cate_labels[2] = 19
cate_labels[3] = 7
cate_labels = paddle.to_tensor(cate_labels, place=place)
cate_labels = L.cast(cate_labels, 'int32')

cate_scores = np.zeros((4, ))
cate_scores[0] = 0.9
cate_scores[1] = 0.8
cate_scores[2] = 0.7
cate_scores[3] = 0.6
cate_scores = paddle.to_tensor(cate_scores, place=place)
cate_scores = L.cast(cate_scores, 'float32')


gaussian_sigma = 2.0
use_gaussian = False

# Matrix NMS
kernel = 'gaussian' if use_gaussian else 'linear'
cate_scores = _matrix_nms(bboxes, cate_labels, cate_scores, kernel=kernel, sigma=gaussian_sigma)

print(cate_scores)








