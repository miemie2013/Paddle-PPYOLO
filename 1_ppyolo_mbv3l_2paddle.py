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
from model.ppyolo import PPYOLO
import paddle.fluid as fluid
import paddle
import os

print(paddle.__version__)
paddle.disable_static()
# 开启动态图



use_gpu = True

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()



cfg = PPYOLO_mobilenet_v3_large_Config()
model_path = 'ppyolo_mobilenet_v3_large.pdparams'



def load_weights(path):
    state_dict = fluid.io.load_program_state(path)
    return state_dict

state_dict = load_weights(model_path)
print('============================================================')

backbone_dic = {}
fpn_dic = {}
head_dic = {}
yolo_dic = {}
yolo_dic2 = {}
yolo_dic3 = {}
others = {}
for key, value in state_dict.items():
    if 'branch' in key:
        backbone_dic[key] = value
    elif 'yolo_block.0' in key:
        yolo_dic[key] = value
    elif 'yolo_block.1' in key:
        yolo_dic2[key] = value
    elif 'yolo' in key:
        yolo_dic3[key] = value
    elif 'conv1_' in key:
        fpn_dic[key] = value
    elif 'conv5_' in key:
        head_dic[key] = value
    else:
        others[key] = value

print()



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



# MobileNetV3
multiplier = 1.0
scale = 1.0
model_name = 'large'
feature_maps = [1, 2, 3, 4, 6]
extra_block_filters = []
conv_decay = 0.0
norm_decay = 0.0
inplanes = 16
end_points = []
block_stride = 0

lr_mult_list = [1.0, 1.0, 1.0, 1.0, 1.0]
freeze_norm = False
norm_type = 'sync_bn'
curr_stage = 0

if model_name == "large":
    cfg = [
        # kernel_size, expand, channel, se_block, act_mode, stride
        [3, 16, 16, False, 'relu', 1],
        [3, 64, 24, False, 'relu', 2],
        [3, 72, 24, False, 'relu', 1],
        [5, 72, 40, True, 'relu', 2],
        [5, 120, 40, True, 'relu', 1],
        [5, 120, 40, True, 'relu', 1],
        [3, 240, 80, False, 'hard_swish', 2],
        [3, 200, 80, False, 'hard_swish', 1],
        [3, 184, 80, False, 'hard_swish', 1],
        [3, 184, 80, False, 'hard_swish', 1],
        [3, 480, 112, True, 'hard_swish', 1],
        [3, 672, 112, True, 'hard_swish', 1],
        [5, 672, 160, True, 'hard_swish', 2],
        [5, 960, 160, True, 'hard_swish', 1],
        [5, 960, 160, True, 'hard_swish', 1],
    ]
    cls_ch_squeeze = 960
    cls_ch_expand = 1280
elif model_name == "small":
    cfg = [
        # kernel_size, expand, channel, se_block, act_mode, stride
        [3, 16, 16, True, 'relu', 2],
        [3, 72, 24, False, 'relu', 2],
        [3, 88, 24, False, 'relu', 1],
        [5, 96, 40, True, 'hard_swish', 2],
        [5, 240, 40, True, 'hard_swish', 1],
        [5, 240, 40, True, 'hard_swish', 1],
        [5, 120, 48, True, 'hard_swish', 1],
        [5, 144, 48, True, 'hard_swish', 1],
        [5, 288, 96, True, 'hard_swish', 2],
        [5, 576, 96, True, 'hard_swish', 1],
        [5, 576, 96, True, 'hard_swish', 1],
    ]
    cls_ch_squeeze = 576
    cls_ch_expand = 1280
else:
    raise NotImplementedError

if multiplier != 1.0:
    cfg[-3][2] = int(cfg[-3][2] * multiplier)
    cfg[-2][1] = int(cfg[-2][1] * multiplier)
    cfg[-2][2] = int(cfg[-2][2] * multiplier)
    cfg[-1][1] = int(cfg[-1][1] * multiplier)
    cfg[-1][2] = int(cfg[-1][2] * multiplier)




def copyResidualUnit(residual_unit_name, use_se=False, name=None,):
    w = state_dict[name + '_expand_weights']
    scale = state_dict[name + '_expand_bn_scale']
    offset = state_dict[name + '_expand_bn_offset']
    m = state_dict[name + '_expand_bn_mean']
    v = state_dict[name + '_expand_bn_variance']
    copy_conv_bn(residual_unit_name+'.expand', w, scale, offset, m, v)

    w = state_dict[name + '_depthwise_weights']
    scale = state_dict[name + '_depthwise_bn_scale']
    offset = state_dict[name + '_depthwise_bn_offset']
    m = state_dict[name + '_depthwise_bn_mean']
    v = state_dict[name + '_depthwise_bn_variance']
    copy_conv_bn(residual_unit_name+'.res_conv1_depthwise', w, scale, offset, m, v)

    if use_se:
        w = state_dict[name + '_se_1_weights']
        b = state_dict[name + '_se_1_offset']
        copy_conv(residual_unit_name+'.se.conv1', w, b)
        w = state_dict[name + '_se_2_weights']
        b = state_dict[name + '_se_2_offset']
        copy_conv(residual_unit_name+'.se.conv2', w, b)

    w = state_dict[name + '_linear_weights']
    scale = state_dict[name + '_linear_bn_scale']
    offset = state_dict[name + '_linear_bn_offset']
    m = state_dict[name + '_linear_bn_mean']
    v = state_dict[name + '_linear_bn_variance']
    copy_conv_bn(residual_unit_name+'.linear', w, scale, offset, m, v)


blocks = []

w = state_dict['conv1_weights']
scale = state_dict['conv1_bn_scale']
offset = state_dict['conv1_bn_offset']
m = state_dict['conv1_bn_mean']
v = state_dict['conv1_bn_variance']
copy_conv_bn('backbone.conv1', w, scale, offset, m, v)


i = 0
for layer_cfg in cfg:
    if layer_cfg[5] == 2:
        block_stride += 1

    copyResidualUnit(
        'backbone._residual_units.%d' % i,
        use_se=layer_cfg[3],
        name='conv' + str(i + 2))

    if block_stride == 4 and layer_cfg[5] == 2:
        block_stride += 1
    i += 1
    curr_stage += 1




# head

conv_block_num = 0
num_classes = 80
anchors = [[10, 14], [23, 27], [37, 58],
           [81, 82], [135, 169], [344, 319]]
anchor_masks = [[3, 4, 5], [0, 1, 2]]
batch_size = 1
norm_type = "bn"
coord_conv = True
iou_aware = False
iou_aware_factor = 0.4
block_size = 3
scale_x_y = 1.05
use_spp = True
drop_block = True
keep_prob = 0.9
clip_bbox = True
yolo_loss = None
downsample = [32, 16]
in_channels = [160, 112]
nms_cfg = None
is_train = False

bn = 0
gn = 0
af = 0
if norm_type == 'bn':
    bn = 1
elif norm_type == 'gn':
    gn = 1
elif norm_type == 'affine_channel':
    af = 1



def copy_DetectionBlock(
        _detection_block,
        param_state_dict,
        i,
        in_c,
             channel,
             coord_conv=True,
             bn=0,
             gn=0,
             af=0,
             conv_block_num=2,
             is_first=False,
             use_spp=True,
             drop_block=True,
             block_size=3,
             keep_prob=0.9,
             is_test=True,
        name=''):
    kkk = 0
    for j in range(conv_block_num):
        kkk += 1

        conv_name = '{}.{}.0'.format(name, j)
        w = state_dict[conv_name + '.conv.weights']
        scale = state_dict[conv_name + '.bn.scale']
        offset = state_dict[conv_name + '.bn.offset']
        m = state_dict[conv_name + '.bn.mean']
        v = state_dict[conv_name + '.bn.var']
        conv_unit_name = 'head.detection_blocks.%d.layers.%d' % (i, kkk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)
        kkk += 1


        if use_spp and is_first and j == 1:
            kkk += 1

            conv_name = '{}.{}.spp.conv'.format(name, j)
            w = state_dict[conv_name + '.conv.weights']
            scale = state_dict[conv_name + '.bn.scale']
            offset = state_dict[conv_name + '.bn.offset']
            m = state_dict[conv_name + '.bn.mean']
            v = state_dict[conv_name + '.bn.var']
            conv_unit_name = 'head.detection_blocks.%d.layers.%d' % (i, kkk)
            copy_conv_bn(conv_unit_name, w, scale, offset, m, v)
            kkk += 1

            conv_name = '{}.{}.1'.format(name, j)
            w = state_dict[conv_name + '.conv.weights']
            scale = state_dict[conv_name + '.bn.scale']
            offset = state_dict[conv_name + '.bn.offset']
            m = state_dict[conv_name + '.bn.mean']
            v = state_dict[conv_name + '.bn.var']
            conv_unit_name = 'head.detection_blocks.%d.layers.%d' % (i, kkk)
            copy_conv_bn(conv_unit_name, w, scale, offset, m, v)
            kkk += 1
        else:
            conv_name = '{}.{}.1'.format(name, j)
            w = state_dict[conv_name + '.conv.weights']
            scale = state_dict[conv_name + '.bn.scale']
            offset = state_dict[conv_name + '.bn.offset']
            m = state_dict[conv_name + '.bn.mean']
            v = state_dict[conv_name + '.bn.var']
            conv_unit_name = 'head.detection_blocks.%d.layers.%d' % (i, kkk)
            copy_conv_bn(conv_unit_name, w, scale, offset, m, v)
            kkk += 1

        if drop_block and j == 0 and not is_first:
            kkk += 1

    if drop_block and is_first:
        kkk += 1

    kkk += 1

    conv_name = '{}.2'.format(name)
    w = state_dict[conv_name + '.conv.weights']
    scale = state_dict[conv_name + '.bn.scale']
    offset = state_dict[conv_name + '.bn.offset']
    m = state_dict[conv_name + '.bn.mean']
    v = state_dict[conv_name + '.bn.var']
    conv_unit_name = 'head.detection_blocks.%d.layers.%d' % (i, kkk)
    copy_conv_bn(conv_unit_name, w, scale, offset, m, v)
    kkk += 1

    conv_name = '{}.tip'.format(name)
    w = state_dict[conv_name + '.conv.weights']
    scale = state_dict[conv_name + '.bn.scale']
    offset = state_dict[conv_name + '.bn.offset']
    m = state_dict[conv_name + '.bn.mean']
    v = state_dict[conv_name + '.bn.var']
    conv_unit_name = 'head.detection_blocks.%d.tip_layers.%d' % (i, 1)
    copy_conv_bn(conv_unit_name, w, scale, offset, m, v)



out_layer_num = len(downsample)
for i in range(out_layer_num):
    copy_DetectionBlock(
        head.detection_blocks[i],
        param_state_dict,
        i,
        in_c=in_channels[i],
        channel=64 * (2**out_layer_num) // (2**i),
        coord_conv=coord_conv,
        bn=bn,
        gn=gn,
        af=af,
        is_first=i == 0,
        conv_block_num=conv_block_num,
        use_spp=use_spp,
        drop_block=drop_block,
        block_size=block_size,
        keep_prob=keep_prob,
        is_test=(not is_train),
        name="yolo_block.{}".format(i)
    )

    w = state_dict["yolo_output.{}.conv.weights".format(i)]
    b = state_dict["yolo_output.{}.conv.bias".format(i)]
    conv_name = 'head.yolo_output_convs.%d.conv' % (i, )
    copy_conv(conv_name, w, b)

    if i < out_layer_num - 1:
        conv_name = "yolo_transition.{}".format(i)
        w = state_dict[conv_name + '.conv.weights']
        scale = state_dict[conv_name + '.bn.scale']
        offset = state_dict[conv_name + '.bn.offset']
        m = state_dict[conv_name + '.bn.mean']
        v = state_dict[conv_name + '.bn.var']
        conv_unit_name = 'head.upsample_layers.%d' % (i*2, )
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


save_name = 'dygraph_ppyolo_mobilenet_v3_large.pdparams'
paddle.save(param_state_dict, save_name)
print('\nDone.')







