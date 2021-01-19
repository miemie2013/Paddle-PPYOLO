#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
from config import *
from model.architectures.fcos import *
import paddle
import torch
import os

print(paddle.__version__)
paddle.disable_static()
# 开启动态图



use_gpu = True

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()



cfg = FCOS_RT_R50_FPN_4x_Config()
# 该模型是COCO数据集上训练好的，所以强制改类别数为80
cfg.num_classes = 80
cfg.head['num_classes'] = cfg.num_classes
model_path = 'FCOS_RT_MS_R_50_4x_syncbn.pth'



def load_weights(path):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights(model_path)
print('============================================================')

backbone_dic = {}
scale_on_reg_dic = {}
fpn_dic = {}
head_dic = {}
others = {}
for key, value in state_dict.items():
    if 'tracked' in key:
        continue
    if 'bottom_up' in key:
        backbone_dic[key] = value.data.numpy()
    elif 'fpn' in key:
        fpn_dic[key] = value.data.numpy()
    elif 'fcos_head' in key:
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
head = Head(fcos_loss=None, nms_cfg=cfg.nms_cfg, **cfg.head)
model = FCOS(backbone, fpn, head)

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

def copy_conv_af(conv_unit_name, w, scale, offset):
    copy(conv_unit_name + '.conv.weight', w)
    copy(conv_unit_name + '.scale', scale)
    copy(conv_unit_name + '.offset', offset)


def copy_conv(conv_name, w, b):
    copy(conv_name + '.weight', w)
    copy(conv_name + '.bias', b)


def copy_conv_gn(conv_unit_name, w, b, scale, offset):
    copy(conv_unit_name + '.conv.weight', w)
    copy(conv_unit_name + '.conv.bias', b)
    copy(conv_unit_name + '.gn.weight', scale)
    copy(conv_unit_name + '.gn.bias', offset)



# Resnet50Vb
# AdelaiDet里输入图片使用了BGR格式。这里做一下手脚使输入图片默认是RGB格式。
w = backbone_dic['backbone.bottom_up.stem.conv1.weight']
cpw = np.copy(w)
w[:, 2, :, :] = cpw[:, 0, :, :]
w[:, 0, :, :] = cpw[:, 2, :, :]
scale = backbone_dic['backbone.bottom_up.stem.conv1.norm.weight']
offset = backbone_dic['backbone.bottom_up.stem.conv1.norm.bias']
m = backbone_dic['backbone.bottom_up.stem.conv1.norm.running_mean']
v = backbone_dic['backbone.bottom_up.stem.conv1.norm.running_var']
copy_conv_bn('backbone.stage1_conv1_1', w, scale, offset, m, v)



nums = [3, 4, 6, 3]
dcn_v2_stages = [3, 4, 5]
for nid, num in enumerate(nums):
    stage_name = 'res' + str(nid + 2)
    for kk in range(num):
        block_name = stage_name + chr(ord("a") + kk)

        conv_name1 = 'backbone.bottom_up.%s.%d.conv1' % (stage_name, kk)
        w = backbone_dic[conv_name1 + '.weight']
        scale = backbone_dic[conv_name1 + '.norm.weight']
        offset = backbone_dic[conv_name1 + '.norm.bias']
        m = backbone_dic[conv_name1 + '.norm.running_mean']
        v = backbone_dic[conv_name1 + '.norm.running_var']
        conv_unit_name = 'backbone.stage%d_%d.conv1' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


        conv_name2 = 'backbone.bottom_up.%s.%d.conv2' % (stage_name, kk)
        w = backbone_dic[conv_name2 + '.weight']
        scale = backbone_dic[conv_name2 + '.norm.weight']
        offset = backbone_dic[conv_name2 + '.norm.bias']
        m = backbone_dic[conv_name2 + '.norm.running_mean']
        v = backbone_dic[conv_name2 + '.norm.running_var']
        conv_unit_name = 'backbone.stage%d_%d.conv2' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


        conv_name3 = 'backbone.bottom_up.%s.%d.conv3' % (stage_name, kk)
        w = backbone_dic[conv_name3 + '.weight']
        scale = backbone_dic[conv_name3 + '.norm.weight']
        offset = backbone_dic[conv_name3 + '.norm.bias']
        m = backbone_dic[conv_name3 + '.norm.running_mean']
        v = backbone_dic[conv_name3 + '.norm.running_var']
        conv_unit_name = 'backbone.stage%d_%d.conv3' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)

        # 每个stage的第一个卷积块才有4个卷积层
        if kk == 0:
            shortcut_name = 'backbone.bottom_up.%s.%d.shortcut' % (stage_name, kk)
            w = backbone_dic[shortcut_name + '.weight']
            scale = backbone_dic[shortcut_name + '.norm.weight']
            offset = backbone_dic[shortcut_name + '.norm.bias']
            m = backbone_dic[shortcut_name + '.norm.running_mean']
            v = backbone_dic[shortcut_name + '.norm.running_var']
            conv_unit_name = 'backbone.stage%d_%d.conv4' % (2+nid, kk)
            copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


# fpn
w = fpn_dic['backbone.fpn_lateral5.weight']
b = fpn_dic['backbone.fpn_lateral5.bias']
copy_conv('fpn.fpn_inner_convs.0.conv', w, b)

w = fpn_dic['backbone.fpn_lateral4.weight']
b = fpn_dic['backbone.fpn_lateral4.bias']
copy_conv('fpn.fpn_inner_convs.1.conv', w, b)

w = fpn_dic['backbone.fpn_lateral3.weight']
b = fpn_dic['backbone.fpn_lateral3.bias']
copy_conv('fpn.fpn_inner_convs.2.conv', w, b)

w = fpn_dic['backbone.fpn_output5.weight']
b = fpn_dic['backbone.fpn_output5.bias']
copy_conv('fpn.fpn_convs.0.conv', w, b)

w = fpn_dic['backbone.fpn_output4.weight']
b = fpn_dic['backbone.fpn_output4.bias']
copy_conv('fpn.fpn_convs.1.conv', w, b)

w = fpn_dic['backbone.fpn_output3.weight']
b = fpn_dic['backbone.fpn_output3.bias']
copy_conv('fpn.fpn_convs.2.conv', w, b)


# head
num_convs = 4
ids = [[0, 1], [3, 4], [6, 7], [9, 10]]
for lvl in range(0, num_convs):
    # conv + gn
    w = head_dic['proposal_generator.fcos_head.cls_tower.%d.weight'%ids[lvl][0]]
    b = head_dic['proposal_generator.fcos_head.cls_tower.%d.bias'%ids[lvl][0]]
    scale = head_dic['proposal_generator.fcos_head.cls_tower.%d.weight'%ids[lvl][1]]
    offset = head_dic['proposal_generator.fcos_head.cls_tower.%d.bias'%ids[lvl][1]]
    copy_conv_gn('head.cls_convs.%d' % (lvl, ), w, b, scale, offset)


    # conv + gn
    w = head_dic['proposal_generator.fcos_head.bbox_tower.%d.weight'%ids[lvl][0]]
    b = head_dic['proposal_generator.fcos_head.bbox_tower.%d.bias'%ids[lvl][0]]
    scale = head_dic['proposal_generator.fcos_head.bbox_tower.%d.weight'%ids[lvl][1]]
    offset = head_dic['proposal_generator.fcos_head.bbox_tower.%d.bias'%ids[lvl][1]]
    copy_conv_gn('head.reg_convs.%d' % (lvl, ), w, b, scale, offset)

# 类别分支最后的conv
w = head_dic['proposal_generator.fcos_head.cls_logits.weight']
b = head_dic['proposal_generator.fcos_head.cls_logits.bias']
copy_conv('head.cls_convs.%d.conv' % (num_convs, ), w, b)

# 坐标分支最后的conv
w = head_dic['proposal_generator.fcos_head.bbox_pred.weight']
b = head_dic['proposal_generator.fcos_head.bbox_pred.bias']
copy_conv('head.reg_convs.%d.conv' % (num_convs, ), w, b)

# centerness分支最后的conv
w = head_dic['proposal_generator.fcos_head.ctrness.weight']
b = head_dic['proposal_generator.fcos_head.ctrness.bias']
copy_conv('head.ctn_conv.conv', w, b)



# 3个scale。请注意，AdelaiDet在head部分是从小感受野到大感受野遍历，而PaddleDetection是从大感受野到小感受野遍历。所以这里scale顺序反过来。
scale_i = head_dic['proposal_generator.fcos_head.scales.0.scale']
copy('head.scales_on_reg.%d'% 2, scale_i)
scale_i = head_dic['proposal_generator.fcos_head.scales.1.scale']
copy('head.scales_on_reg.%d'% 1, scale_i)
scale_i = head_dic['proposal_generator.fcos_head.scales.2.scale']
copy('head.scales_on_reg.%d'% 0, scale_i)



save_name = 'dygraph_fcos_rt_r50_fpn_4x.pdparams'
paddle.save(param_state_dict, save_name)
print('\nDone.')







