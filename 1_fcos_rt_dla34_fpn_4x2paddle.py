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



cfg = FCOS_RT_DLA34_FPN_4x_Config()
# 该模型是COCO数据集上训练好的，所以强制改类别数为80
cfg.num_classes = 80
cfg.head['num_classes'] = cfg.num_classes
model_path = 'FCOS_RT_MS_DLA_34_4x_syncbn.pth'



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



# dla34
# AdelaiDet里输入图片使用了BGR格式。这里做一下手脚使输入图片默认是RGB格式。
w = backbone_dic['backbone.bottom_up.base_layer.0.weight']
cpw = np.copy(w)
w[:, 2, :, :] = cpw[:, 0, :, :]
w[:, 0, :, :] = cpw[:, 2, :, :]
scale = backbone_dic['backbone.bottom_up.base_layer.1.weight']
offset = backbone_dic['backbone.bottom_up.base_layer.1.bias']
m = backbone_dic['backbone.bottom_up.base_layer.1.running_mean']
v = backbone_dic['backbone.bottom_up.base_layer.1.running_var']
copy_conv_bn('backbone.base_layer', w, scale, offset, m, v)

w = backbone_dic['backbone.bottom_up.level0.0.weight']
scale = backbone_dic['backbone.bottom_up.level0.1.weight']
offset = backbone_dic['backbone.bottom_up.level0.1.bias']
m = backbone_dic['backbone.bottom_up.level0.1.running_mean']
v = backbone_dic['backbone.bottom_up.level0.1.running_var']
copy_conv_bn('backbone.level0.0', w, scale, offset, m, v)


w = backbone_dic['backbone.bottom_up.level1.0.weight']
scale = backbone_dic['backbone.bottom_up.level1.1.weight']
offset = backbone_dic['backbone.bottom_up.level1.1.bias']
m = backbone_dic['backbone.bottom_up.level1.1.running_mean']
v = backbone_dic['backbone.bottom_up.level1.1.running_var']
copy_conv_bn('backbone.level1.0', w, scale, offset, m, v)


def copy_Tree(tree_name, levels, in_channels, out_channels, name=''):
    if levels == 1:
        w = backbone_dic[name + '.tree1.conv1.weight']
        scale = backbone_dic[name + '.tree1.bn1.weight']
        offset = backbone_dic[name + '.tree1.bn1.bias']
        m = backbone_dic[name + '.tree1.bn1.running_mean']
        v = backbone_dic[name + '.tree1.bn1.running_var']
        copy_conv_bn(tree_name+'.tree1.conv1', w, scale, offset, m, v)

        w = backbone_dic[name + '.tree1.conv2.weight']
        scale = backbone_dic[name + '.tree1.bn2.weight']
        offset = backbone_dic[name + '.tree1.bn2.bias']
        m = backbone_dic[name + '.tree1.bn2.running_mean']
        v = backbone_dic[name + '.tree1.bn2.running_var']
        copy_conv_bn(tree_name+'.tree1.conv2', w, scale, offset, m, v)

        w = backbone_dic[name + '.tree2.conv1.weight']
        scale = backbone_dic[name + '.tree2.bn1.weight']
        offset = backbone_dic[name + '.tree2.bn1.bias']
        m = backbone_dic[name + '.tree2.bn1.running_mean']
        v = backbone_dic[name + '.tree2.bn1.running_var']
        copy_conv_bn(tree_name+'.tree2.conv1', w, scale, offset, m, v)

        w = backbone_dic[name + '.tree2.conv2.weight']
        scale = backbone_dic[name + '.tree2.bn2.weight']
        offset = backbone_dic[name + '.tree2.bn2.bias']
        m = backbone_dic[name + '.tree2.bn2.running_mean']
        v = backbone_dic[name + '.tree2.bn2.running_var']
        copy_conv_bn(tree_name+'.tree2.conv2', w, scale, offset, m, v)
    else:
        copy_Tree(tree_name+'.tree1', levels - 1, in_channels, out_channels, name=name+'.tree1')
        copy_Tree(tree_name+'.tree2', levels - 1, out_channels, out_channels, name=name+'.tree2')
    if levels == 1:
        w = backbone_dic[name + '.root.conv.weight']
        scale = backbone_dic[name + '.root.bn.weight']
        offset = backbone_dic[name + '.root.bn.bias']
        m = backbone_dic[name + '.root.bn.running_mean']
        v = backbone_dic[name + '.root.bn.running_var']
        copy_conv_bn(tree_name+'.root.conv', w, scale, offset, m, v)
    if in_channels != out_channels:
        w = backbone_dic[name + '.project.0.weight']
        scale = backbone_dic[name + '.project.1.weight']
        offset = backbone_dic[name + '.project.1.bias']
        m = backbone_dic[name + '.project.1.running_mean']
        v = backbone_dic[name + '.project.1.running_var']
        copy_conv_bn(tree_name+'.project', w, scale, offset, m, v)


levels = [1, 1, 1, 2, 2, 1]
channels = [16, 32, 64, 128, 256, 512]

copy_Tree('backbone.level2', levels[2], channels[1], channels[2], 'backbone.bottom_up.level2')
copy_Tree('backbone.level3', levels[3], channels[2], channels[3], 'backbone.bottom_up.level3')
copy_Tree('backbone.level4', levels[4], channels[3], channels[4], 'backbone.bottom_up.level4')
copy_Tree('backbone.level5', levels[5], channels[4], channels[5], 'backbone.bottom_up.level5')


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



save_name = 'dygraph_fcos_rt_dla34_fpn_4x.pdparams'
paddle.save(param_state_dict, save_name)
print('\nDone.')







