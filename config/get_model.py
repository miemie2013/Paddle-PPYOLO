#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-23 09:13:23
#   Description : paddle2.0_ppyolo
#
# ================================================================
from model.backbones.resnet_vb import *
from model.backbones.resnet_vd import *
from model.backbones.cspdarknet import *
from model.backbones.mobilenet_v3 import *
from model.backbones.dla import *
from model.backbones.fpn import *

from model.anchor_heads.yolov3_head import *
from model.anchor_heads.yolov4_head import *
from model.anchor_heads.fcos_head import *
from model.anchor_heads.maskfeat_head import *
from model.anchor_heads.solov2_head import *
from model.anchor_heads.reppoints_head import *

from model.losses.yolov3_loss import *
from model.losses.my_loss import *
from model.losses.iou_losses import *
from model.losses.fcos_loss import *
from model.losses.solov2_loss import *


def select_backbone(name):
    if name == 'Resnet50Vd':
        return Resnet50Vd
    if name == 'Resnet50Vb':
        return Resnet50Vb
    if name == 'Resnet18Vd':
        return Resnet18Vd
    if name == 'MobileNetV3':
        return MobileNetV3
    if name == 'CSPDarknet53':
        return CSPDarknet53
    if name == 'DLA':
        return DLA

def select_head(name):
    if name == 'YOLOv3Head':
        return YOLOv3Head
    if name == 'YOLOv4Head':
        return YOLOv4Head
    if name == 'FCOSHead':
        return FCOSHead
    if name == 'SOLOv2Head':
        return SOLOv2Head
    if name == 'MaskFeatHead':
        return MaskFeatHead
    if name == 'RepPointsHead':
        return RepPointsHead

def select_fpn(name):
    if name == 'FPN':
        return FPN

def select_loss(name):
    if name == 'YOLOv3Loss':
        return YOLOv3Loss
    if name == 'IouLoss':
        return IouLoss
    if name == 'IouAwareLoss':
        return IouAwareLoss
    if name == 'MyLoss':
        return MyLoss
    if name == 'FCOSLoss':
        return FCOSLoss
    if name == 'SOLOv2Loss':
        return SOLOv2Loss

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




