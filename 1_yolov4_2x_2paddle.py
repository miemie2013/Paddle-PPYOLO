#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-23 09:13:23
#   Description : paddle2.0_ppyolo
#
# ================================================================
from config import *
from model.architectures.yolo import PPYOLO
import paddle.fluid as fluid
import paddle
import torch
import os

print(paddle.__version__)
paddle.disable_static()
# 开启动态图



use_gpu = True

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()



cfg = YOLOv4_2x_Config()
# 该模型是COCO数据集上训练好的，所以强制改类别数为80
cfg.num_classes = 80
cfg.head['num_classes'] = cfg.num_classes
model_path = 'yolov4.pt'



def load_weights(path):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights(model_path)
print('============================================================')




# 创建模型
Backbone = select_backbone(cfg.backbone_type)
backbone = Backbone(**cfg.backbone)
Head = select_head(cfg.head_type)
head = Head(yolo_loss=None, nms_cfg=cfg.nms_cfg, **cfg.head)
ppyolo = PPYOLO(backbone, head)

ppyolo.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式。
param_state_dict = ppyolo.state_dict()

print('\nCopying...')



def copy(name, w):
    value2 = paddle.to_tensor(w, place=place)
    value = param_state_dict[name]
    value = value * 0 + value2
    param_state_dict[name] = value

def copy_conv_bn(conv_unit_name, w, scale, offset, m, v):
    copy(conv_unit_name + '.conv.weight', w)
    copy(conv_unit_name + '.bn.weight', scale)
    copy(conv_unit_name + '.bn.bias', offset)
    copy(conv_unit_name + '.bn._mean', m)
    copy(conv_unit_name + '.bn._variance', v)


def copy_conv(conv_name, w, b):
    copy(conv_name + '.weight', w)
    copy(conv_name + '.bias', b)


def get(idx):
    keyword1 = 'conv%d.weight' % idx
    keyword2 = 'bn%d.weight' % idx
    keyword3 = 'bn%d.bias' % idx
    keyword4 = 'bn%d.running_mean' % idx
    keyword5 = 'bn%d.running_var' % idx
    w, scale, offset, m, v = None, None, None, None, None
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            scale = value
        elif keyword3 in key:
            offset = value
        elif keyword4 in key:
            m = value
        elif keyword5 in key:
            v = value
    return w, scale, offset, m, v

def get2(idx):
    keyword1 = 'conv%d.weight' % idx
    keyword2 = 'conv%d.bias' % idx
    w, b = None, None
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            b = value
    return w, b


def copy_blocks(blocks_name, k, n):
    for i in range(n):
        w, scale, offset, m, v = get(k+i*2)
        copy_conv_bn('%s.sequential.%d.conv1' % (blocks_name, i), w, scale, offset, m, v)
        w, scale, offset, m, v = get(k+i*2+1)
        copy_conv_bn('%s.sequential.%d.conv2' % (blocks_name, i), w, scale, offset, m, v)



# CSPDarknet53
k = 1
w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.conv1', w, scale, offset, m, v)

# stage1
w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage1_conv1', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage1_conv2', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage1_conv3', w, scale, offset, m, v)

copy_blocks('backbone.stage1_blocks', k, n=1)
k += 2*1

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage1_conv4', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage1_conv5', w, scale, offset, m, v)

# stage2
w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage2_conv1', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage2_conv2', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage2_conv3', w, scale, offset, m, v)

copy_blocks('backbone.stage2_blocks', k, n=2)
k += 2*2

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage2_conv4', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage2_conv5', w, scale, offset, m, v)

# stage3
w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage3_conv1', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage3_conv2', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage3_conv3', w, scale, offset, m, v)

copy_blocks('backbone.stage3_blocks', k, n=8)
k += 2*8

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage3_conv4', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage3_conv5', w, scale, offset, m, v)

# stage4
w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage4_conv1', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage4_conv2', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage4_conv3', w, scale, offset, m, v)

copy_blocks('backbone.stage4_blocks', k, n=8)
k += 2*8

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage4_conv4', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage4_conv5', w, scale, offset, m, v)

# stage5
w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage5_conv1', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage5_conv2', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage5_conv3', w, scale, offset, m, v)

copy_blocks('backbone.stage5_blocks', k, n=4)
k += 2*4

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage5_conv4', w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn('backbone.stage5_conv5', w, scale, offset, m, v)


# head

for i in range(73, 94, 1):
    w, scale, offset, m, v = get(i)
    copy_conv_bn('head.conv%.3d' % i, w, scale, offset, m, v)
for i in range(95, 102, 1):
    w, scale, offset, m, v = get(i)
    copy_conv_bn('head.conv%.3d' % i, w, scale, offset, m, v)
for i in range(103, 110, 1):
    w, scale, offset, m, v = get(i)
    copy_conv_bn('head.conv%.3d' % i, w, scale, offset, m, v)


w, b = get2(94)
copy_conv('head.conv094.conv', w, b)
w, b = get2(102)
copy_conv('head.conv102.conv', w, b)
w, b = get2(110)
copy_conv('head.conv110.conv', w, b)


save_name = 'dygraph_yolov4_2x.pdparams'
paddle.save(param_state_dict, save_name)
print('\nDone.')







