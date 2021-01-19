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
import os

print(paddle.__version__)
paddle.disable_static()
# 开启动态图



use_gpu = True

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()



cfg = FCOS_R50_FPN_Multiscale_2x_Config()
# 该模型是COCO数据集上训练好的，所以强制改类别数为80
cfg.num_classes = 80
cfg.head['num_classes'] = cfg.num_classes
model_path = 'fcos_r50_fpn_multiscale_2x.pdparams'



def load_weights(path):
    state_dict = fluid.io.load_program_state(path)
    return state_dict

state_dict = load_weights(model_path)
print('============================================================')

backbone_dic = {}
scale_on_reg_dic = {}
fpn_dic = {}
head_dic = {}
others = {}
for key, value in state_dict.items():
    # if 'tracked' in key:
    #     continue
    if 'branch' in key:
        backbone_dic[key] = value
    elif 'scale_on_reg' in key:
        scale_on_reg_dic[key] = value
    elif 'fpn' in key:
        fpn_dic[key] = value
    elif 'fcos_head' in key:
        head_dic[key] = value
    else:
        others[key] = value

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
w = state_dict['conv1_weights']
scale = state_dict['bn_conv1_scale']
offset = state_dict['bn_conv1_offset']
copy_conv_af('backbone.stage1_conv1_1', w, scale, offset)



nums = [3, 4, 6, 3]
dcn_v2_stages = [3, 4, 5]
for nid, num in enumerate(nums):
    stage_name = 'res' + str(nid + 2)
    for kk in range(num):
        block_name = stage_name + chr(ord("a") + kk)

        conv_name1 = block_name + "_branch2a"
        bn_name1 = 'bn' + conv_name1[3:]
        w = backbone_dic[conv_name1 + '_weights']
        scale = backbone_dic[bn_name1 + '_scale']
        offset = backbone_dic[bn_name1 + '_offset']
        conv_unit_name = 'backbone.stage%d_%d.conv1' % (2+nid, kk)
        copy_conv_af(conv_unit_name, w, scale, offset)


        conv_name2 = block_name + "_branch2b"
        bn_name2 = 'bn' + conv_name2[3:]
        w = state_dict[conv_name2 + '_weights']
        scale = state_dict[bn_name2 + '_scale']
        offset = state_dict[bn_name2 + '_offset']
        conv_unit_name = 'backbone.stage%d_%d.conv2' % (2+nid, kk)
        copy_conv_af(conv_unit_name, w, scale, offset)


        conv_name3 = block_name + "_branch2c"
        bn_name3 = 'bn' + conv_name3[3:]
        w = backbone_dic[conv_name3 + '_weights']
        scale = backbone_dic[bn_name3 + '_scale']
        offset = backbone_dic[bn_name3 + '_offset']
        conv_unit_name = 'backbone.stage%d_%d.conv3' % (2+nid, kk)
        copy_conv_af(conv_unit_name, w, scale, offset)

        # 每个stage的第一个卷积块才有4个卷积层
        if kk == 0:
            shortcut_name = block_name + "_branch1"
            shortcut_bn_name = 'bn' + shortcut_name[3:]
            w = backbone_dic[shortcut_name + '_weights']
            scale = backbone_dic[shortcut_bn_name + '_scale']
            offset = backbone_dic[shortcut_bn_name + '_offset']
            conv_unit_name = 'backbone.stage%d_%d.conv4' % (2+nid, kk)
            copy_conv_af(conv_unit_name, w, scale, offset)


# fpn
w = fpn_dic['fpn_inner_res5_sum_w']
b = fpn_dic['fpn_inner_res5_sum_b']
copy_conv('fpn.fpn_inner_convs.0.conv', w, b)

w = fpn_dic['fpn_inner_res4_sum_lateral_w']
b = fpn_dic['fpn_inner_res4_sum_lateral_b']
copy_conv('fpn.fpn_inner_convs.1.conv', w, b)

w = fpn_dic['fpn_inner_res3_sum_lateral_w']
b = fpn_dic['fpn_inner_res3_sum_lateral_b']
copy_conv('fpn.fpn_inner_convs.2.conv', w, b)

w = fpn_dic['fpn_res5_sum_w']
b = fpn_dic['fpn_res5_sum_b']
copy_conv('fpn.fpn_convs.0.conv', w, b)

w = fpn_dic['fpn_res4_sum_w']
b = fpn_dic['fpn_res4_sum_b']
copy_conv('fpn.fpn_convs.1.conv', w, b)

w = fpn_dic['fpn_res3_sum_w']
b = fpn_dic['fpn_res3_sum_b']
copy_conv('fpn.fpn_convs.2.conv', w, b)

w = fpn_dic['fpn_6_w']
b = fpn_dic['fpn_6_b']
copy_conv('fpn.extra_convs.0.conv', w, b)

w = fpn_dic['fpn_7_w']
b = fpn_dic['fpn_7_b']
copy_conv('fpn.extra_convs.1.conv', w, b)


# head
num_convs = 4
for lvl in range(0, num_convs):
    # conv + gn
    conv_cls_name = 'fcos_head_cls_tower_conv_{}'.format(lvl)
    norm_name = conv_cls_name + "_norm"
    w = head_dic[conv_cls_name + "_weights"]
    b = head_dic[conv_cls_name + "_bias"]
    scale = head_dic[norm_name + "_scale"]
    offset = head_dic[norm_name + "_offset"]
    copy_conv_gn('head.cls_convs.%d' % (lvl, ), w, b, scale, offset)


    # conv + gn
    conv_reg_name = 'fcos_head_reg_tower_conv_{}'.format(lvl)
    norm_name = conv_reg_name + "_norm"
    w = head_dic[conv_reg_name + "_weights"]
    b = head_dic[conv_reg_name + "_bias"]
    scale = head_dic[norm_name + "_scale"]
    offset = head_dic[norm_name + "_offset"]
    copy_conv_gn('head.reg_convs.%d' % (lvl, ), w, b, scale, offset)

# 类别分支最后的conv
conv_cls_name = "fcos_head_cls"
w = head_dic[conv_cls_name + "_weights"]
b = head_dic[conv_cls_name + "_bias"]
copy_conv('head.cls_convs.%d.conv' % (num_convs, ), w, b)

# 坐标分支最后的conv
conv_reg_name = "fcos_head_reg"
w = head_dic[conv_reg_name + "_weights"]
b = head_dic[conv_reg_name + "_bias"]
copy_conv('head.reg_convs.%d.conv' % (num_convs, ), w, b)

# centerness分支最后的conv
conv_centerness_name = "fcos_head_centerness"
w = head_dic[conv_centerness_name + "_weights"]
b = head_dic[conv_centerness_name + "_bias"]
copy_conv('head.ctn_conv.conv', w, b)



# 5个scale
fpn_names = ['fpn_7', 'fpn_6', 'fpn_res5_sum', 'fpn_res4_sum', 'fpn_res3_sum']
i = 0
for fpn_name in fpn_names:
    scale_i = scale_on_reg_dic["%s_scale_on_reg" % fpn_name]
    copy('head.scales_on_reg.%d'% i, scale_i)
    i += 1



save_name = 'dygraph_fcos_r50_fpn_multiscale_2x.pdparams'
paddle.save(param_state_dict, save_name)
print('\nDone.')







