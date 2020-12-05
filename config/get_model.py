#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-23 09:13:23
#   Description : paddle2.0_ppyolo
#
# ================================================================
from model.losses import *
from model.iou_losses import *
from model.head import *
from model.resnet_vd import *
from model.mobilenet_v3 import *


def select_backbone(name):
    if name == 'Resnet50Vd':
        return Resnet50Vd
    if name == 'Resnet18Vd':
        return Resnet18Vd
    if name == 'MobileNetV3':
        return MobileNetV3

def select_head(name):
    if name == 'YOLOv3Head':
        return YOLOv3Head

def select_loss(name):
    if name == 'YOLOv3Loss':
        return YOLOv3Loss
    if name == 'IouLoss':
        return IouLoss
    if name == 'IouAwareLoss':
        return IouAwareLoss

def select_regularization(name):
    if name == 'L1Decay':
        return fluid.regularizer.L1Decay
    if name == 'L2Decay':
        return fluid.regularizer.L2Decay

def select_optimizer(name):
    if name == 'Momentum':
        return paddle.optimizer.Momentum
    if name == 'Adam':
        return paddle.optimizer.Adam
    if name == 'SGD':
        return paddle.optimizer.SGD




