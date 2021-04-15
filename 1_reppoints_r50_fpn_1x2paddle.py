#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2021-04-15 15:01:23
#   Description :
#
# ================================================================
from config import *
from model.architectures.reppoints import *
import torch
import paddle
import os

print(paddle.__version__)
paddle.disable_static()
# 开启动态图



use_gpu = True

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()



cfg = RepPoints_moment_r50_fpn_1x_Config()
# 该模型是COCO数据集上训练好的，所以强制改类别数为80
cfg.num_classes = 80
cfg.head['num_classes'] = cfg.num_classes



def load_weights(path):
    """ Loads weights from a compressed save file. """
    # state_dict = torch.load(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights('reppoints_moment_r50_fpn_gn-neck+head_1x_coco_20200329-4b38409a.pth')
print('============================================================')

backbone_dic = {}
fpn_dic = {}
head_dic = {}
others = {}
for key, value in state_dict['state_dict'].items():
    if 'tracked' in key:
        continue
    if 'backbone' in key:
        backbone_dic[key] = value.data.numpy()
    elif 'neck' in key:
        fpn_dic[key] = value.data.numpy()
    elif 'bbox_head' in key:
        head_dic[key] = value.data.numpy()
    else:
        others[key] = value.data.numpy()

print()




# 创建模型
Backbone = select_backbone(cfg.backbone_type)
backbone = Backbone(**cfg.backbone)
FPN = select_fpn(cfg.fpn_type)
fpn = FPN(**cfg.fpn)
Head = select_head(cfg.head_type)
head = Head(reppoints_loss=None, nms_cfg=cfg.nms_cfg, **cfg.head)
model = RepPoints(backbone, fpn, head)

model.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式。
param_state_dict = model.state_dict()

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


def copy_conv_gn(conv_unit_name, w, scale, offset):
    copy(conv_unit_name + '.conv.weight', w)
    copy(conv_unit_name + '.gn.weight', scale)
    copy(conv_unit_name + '.gn.bias', offset)



# Resnet50Vb
w = backbone_dic['backbone.conv1.weight']
scale = backbone_dic['backbone.bn1.weight']
offset = backbone_dic['backbone.bn1.bias']
m = backbone_dic['backbone.bn1.running_mean']
v = backbone_dic['backbone.bn1.running_var']
copy_conv_bn('backbone.stage1_conv1_1', w, scale, offset, m, v)



nums = [3, 4, 6, 3]
for nid, num in enumerate(nums):
    stage_name = 'res' + str(nid + 2)
    for kk in range(num):
        block_name = stage_name + chr(ord("a") + kk)

        conv_name1 = 'backbone.layer%d.%d' % ((nid+1), kk)
        w = backbone_dic[conv_name1 + '.conv1.weight']
        scale = backbone_dic[conv_name1 + '.bn1.weight']
        offset = backbone_dic[conv_name1 + '.bn1.bias']
        m = backbone_dic[conv_name1 + '.bn1.running_mean']
        v = backbone_dic[conv_name1 + '.bn1.running_var']
        conv_unit_name = 'backbone.stage%d_%d.conv1' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


        conv_name2 = 'backbone.layer%d.%d' % ((nid+1), kk)
        w = backbone_dic[conv_name2 + '.conv2.weight']
        scale = backbone_dic[conv_name2 + '.bn2.weight']
        offset = backbone_dic[conv_name2 + '.bn2.bias']
        m = backbone_dic[conv_name2 + '.bn2.running_mean']
        v = backbone_dic[conv_name2 + '.bn2.running_var']
        conv_unit_name = 'backbone.stage%d_%d.conv2' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


        conv_name3 = 'backbone.layer%d.%d' % ((nid+1), kk)
        w = backbone_dic[conv_name3 + '.conv3.weight']
        scale = backbone_dic[conv_name3 + '.bn3.weight']
        offset = backbone_dic[conv_name3 + '.bn3.bias']
        m = backbone_dic[conv_name3 + '.bn3.running_mean']
        v = backbone_dic[conv_name3 + '.bn3.running_var']
        conv_unit_name = 'backbone.stage%d_%d.conv3' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)

        # 每个stage的第一个卷积块才有4个卷积层
        if kk == 0:
            shortcut_name = 'backbone.layer%d.%d.downsample' % ((nid + 1), kk)
            w = backbone_dic[shortcut_name + '.0.weight']
            scale = backbone_dic[shortcut_name + '.1.weight']
            offset = backbone_dic[shortcut_name + '.1.bias']
            m = backbone_dic[shortcut_name + '.1.running_mean']
            v = backbone_dic[shortcut_name + '.1.running_var']
            conv_unit_name = 'backbone.stage%d_%d.conv4' % (2+nid, kk)
            copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


# fpn
w = fpn_dic['neck.lateral_convs.2.conv.weight']
scale = fpn_dic['neck.lateral_convs.2.gn.weight']
offset = fpn_dic['neck.lateral_convs.2.gn.bias']
copy_conv_gn('fpn.fpn_inner_convs.0', w, scale, offset)

w = fpn_dic['neck.lateral_convs.1.conv.weight']
scale = fpn_dic['neck.lateral_convs.1.gn.weight']
offset = fpn_dic['neck.lateral_convs.1.gn.bias']
copy_conv_gn('fpn.fpn_inner_convs.1', w, scale, offset)

w = fpn_dic['neck.lateral_convs.0.conv.weight']
scale = fpn_dic['neck.lateral_convs.0.gn.weight']
offset = fpn_dic['neck.lateral_convs.0.gn.bias']
copy_conv_gn('fpn.fpn_inner_convs.2', w, scale, offset)


w = fpn_dic['neck.fpn_convs.0.conv.weight']
scale = fpn_dic['neck.fpn_convs.0.gn.weight']
offset = fpn_dic['neck.fpn_convs.0.gn.bias']
copy_conv_gn('fpn.fpn_convs.2', w, scale, offset)

w = fpn_dic['neck.fpn_convs.1.conv.weight']
scale = fpn_dic['neck.fpn_convs.1.gn.weight']
offset = fpn_dic['neck.fpn_convs.1.gn.bias']
copy_conv_gn('fpn.fpn_convs.1', w, scale, offset)

w = fpn_dic['neck.fpn_convs.2.conv.weight']
scale = fpn_dic['neck.fpn_convs.2.gn.weight']
offset = fpn_dic['neck.fpn_convs.2.gn.bias']
copy_conv_gn('fpn.fpn_convs.0', w, scale, offset)

w = fpn_dic['neck.fpn_convs.3.conv.weight']
scale = fpn_dic['neck.fpn_convs.3.gn.weight']
offset = fpn_dic['neck.fpn_convs.3.gn.bias']
copy_conv_gn('fpn.extra_convs.0', w, scale, offset)

w = fpn_dic['neck.fpn_convs.4.conv.weight']
scale = fpn_dic['neck.fpn_convs.4.gn.weight']
offset = fpn_dic['neck.fpn_convs.4.gn.bias']
copy_conv_gn('fpn.extra_convs.1', w, scale, offset)


# head
num_convs = cfg.head['num_convs']
for lvl in range(0, num_convs):
    # conv + gn
    w = head_dic['bbox_head.reg_convs.%d.conv.weight'%lvl]
    scale = head_dic['bbox_head.reg_convs.%d.gn.weight'%lvl]
    offset = head_dic['bbox_head.reg_convs.%d.gn.bias'%lvl]
    copy_conv_gn('head.reg_convs.%d' % (lvl, ), w, scale, offset)

    # conv + gn
    w = head_dic['bbox_head.cls_convs.%d.conv.weight'%lvl]
    scale = head_dic['bbox_head.cls_convs.%d.gn.weight'%lvl]
    offset = head_dic['bbox_head.cls_convs.%d.gn.bias'%lvl]
    copy_conv_gn('head.cls_convs.%d' % (lvl, ), w, scale, offset)

# 类别分支最后的conv
w = head_dic['bbox_head.reppoints_cls_conv.weight']
copy('head.reppoints_cls_conv_w', w)

w = head_dic['bbox_head.reppoints_cls_out.weight']
b = head_dic['bbox_head.reppoints_cls_out.bias']
copy_conv('head.reppoints_cls_out.conv', w, b)



# 点分支最后的conv
w = head_dic['bbox_head.reppoints_pts_init_conv.weight']
b = head_dic['bbox_head.reppoints_pts_init_conv.bias']
copy_conv('head.reppoints_pts_init_conv.conv', w, b)

w = head_dic['bbox_head.reppoints_pts_init_out.weight']
b = head_dic['bbox_head.reppoints_pts_init_out.bias']
copy_conv('head.reppoints_pts_init_out.conv', w, b)

w = head_dic['bbox_head.reppoints_pts_refine_conv.weight']
copy('head.reppoints_pts_refine_conv_w', w)

w = head_dic['bbox_head.reppoints_pts_refine_out.weight']
b = head_dic['bbox_head.reppoints_pts_refine_out.bias']
copy_conv('head.reppoints_pts_refine_out.conv', w, b)


# 全局学习的系数
w = head_dic['bbox_head.moment_transfer']
copy('head.moment_transfer', w)



save_name = 'dygraph_reppoints_r50_fpn_1x.pdparams'
paddle.save(param_state_dict, save_name)
print('\nDone.')







