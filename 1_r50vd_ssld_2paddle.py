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
from model.architectures.yolo import *
import paddle.fluid as fluid
import paddle
import os

print(paddle.__version__)
paddle.disable_static()
# 开启动态图



use_gpu = True

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()


backbone_cfg = dict(
    norm_type='bn',
    feature_maps=[3, 4, 5],
    dcn_v2_stages=[],
    downsample_in3x3=True,  # 注意这个细节，是在3x3卷积层下采样的。
    freeze_at=0,
    fix_bn_mean_var_at=0,
    freeze_norm=False,
    norm_decay=0.,
)
model_path = 'ResNet50_vd_ssld_pretrained'



def load_weights(path):
    state_dict = fluid.io.load_program_state(path)
    return state_dict

state_dict = load_weights(model_path)
print('============================================================')




# 创建模型
Backbone = select_backbone('Resnet50Vd')
backbone = Backbone(**backbone_cfg)
ppyolo = PPYOLO(backbone, head=None)

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



# Resnet50Vd
w = state_dict['conv1_1_weights']
scale = state_dict['bnv1_1_scale']
offset = state_dict['bnv1_1_offset']
m = state_dict['bnv1_1_mean']
v = state_dict['bnv1_1_variance']
copy_conv_bn('backbone.stage1_conv1_1', w, scale, offset, m, v)

w = state_dict['conv1_2_weights']
scale = state_dict['bnv1_2_scale']
offset = state_dict['bnv1_2_offset']
m = state_dict['bnv1_2_mean']
v = state_dict['bnv1_2_variance']
copy_conv_bn('backbone.stage1_conv1_2', w, scale, offset, m, v)

w = state_dict['conv1_3_weights']
scale = state_dict['bnv1_3_scale']
offset = state_dict['bnv1_3_offset']
m = state_dict['bnv1_3_mean']
v = state_dict['bnv1_3_variance']
copy_conv_bn('backbone.stage1_conv1_3', w, scale, offset, m, v)


nums = [3, 4, 6, 3]
for nid, num in enumerate(nums):
    stage_name = 'res' + str(nid + 2)
    for kk in range(num):
        block_name = stage_name + chr(ord("a") + kk)
        conv_name1 = block_name + "_branch2a"
        conv_name2 = block_name + "_branch2b"
        conv_name3 = block_name + "_branch2c"
        shortcut_name = block_name + "_branch1"

        bn_name1 = 'bn' + conv_name1[3:]
        bn_name2 = 'bn' + conv_name2[3:]
        bn_name3 = 'bn' + conv_name3[3:]
        shortcut_bn_name = 'bn' + shortcut_name[3:]


        w = state_dict[conv_name1 + '_weights']
        scale = state_dict[bn_name1 + '_scale']
        offset = state_dict[bn_name1 + '_offset']
        m = state_dict[bn_name1 + '_mean']
        v = state_dict[bn_name1 + '_variance']
        conv_unit_name = 'backbone.stage%d_%d.conv1' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)

        if (nid+2) in backbone_cfg['dcn_v2_stages']:   # DCNv2
            w = state_dict[conv_name2 + '_weights']
            scale = state_dict[bn_name2 + '_scale']
            offset = state_dict[bn_name2 + '_offset']
            m = state_dict[bn_name2 + '_mean']
            v = state_dict[bn_name2 + '_variance']
            conv_unit_name = 'backbone.stage%d_%d.conv2' % (2+nid, kk)
            copy(conv_unit_name + '.dcn_param', w)
            copy(conv_unit_name + '.bn.weight', scale)
            copy(conv_unit_name + '.bn.bias', offset)
            copy(conv_unit_name + '.bn._mean', m)
            copy(conv_unit_name + '.bn._variance', v)
        else:
            w = state_dict[conv_name2 + '_weights']
            scale = state_dict[bn_name2 + '_scale']
            offset = state_dict[bn_name2 + '_offset']
            m = state_dict[bn_name2 + '_mean']
            v = state_dict[bn_name2 + '_variance']
            conv_unit_name = 'backbone.stage%d_%d.conv2' % (2+nid, kk)
            copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


        w = state_dict[conv_name3 + '_weights']
        scale = state_dict[bn_name3 + '_scale']
        offset = state_dict[bn_name3 + '_offset']
        m = state_dict[bn_name3 + '_mean']
        v = state_dict[bn_name3 + '_variance']
        conv_unit_name = 'backbone.stage%d_%d.conv3' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)

        # 每个stage的第一个卷积块才有4个卷积层
        if kk == 0:
            w = state_dict[shortcut_name + '_weights']
            scale = state_dict[shortcut_bn_name + '_scale']
            offset = state_dict[shortcut_bn_name + '_offset']
            m = state_dict[shortcut_bn_name + '_mean']
            v = state_dict[shortcut_bn_name + '_variance']
            conv_unit_name = 'backbone.stage%d_%d.conv4' % (2+nid, kk)
            copy_conv_bn(conv_unit_name, w, scale, offset, m, v)



save_name = 'dygraph_r50vd_ssld.pdparams'
paddle.save(param_state_dict, save_name)
print('\nDone.')







