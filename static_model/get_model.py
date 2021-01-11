#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
from static_model.resnet_vd import *
from static_model.yolov3_head import *


def select_backbone(name):
    if name == 'Resnet50Vd':
        return Resnet50Vd
    if name == 'Resnet50Vb':
        return Resnet50Vd
    if name == 'Resnet18Vd':
        return Resnet18Vd
    if name == 'MobileNetV3':
        return Resnet50Vd
    if name == 'CSPDarknet53':
        return Resnet50Vd

def select_head(name):
    if name == 'YOLOv3Head':
        return YOLOv3Head
    if name == 'YOLOv4Head':
        return YOLOv3Head




