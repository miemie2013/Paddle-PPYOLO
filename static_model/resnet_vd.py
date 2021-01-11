#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
from static_model.custom_layers import *



class ConvBlock(object):
    def __init__(self, in_c, filters, norm_type, freeze_norm, norm_decay, lr, use_dcn=False, stride=2, downsample_in3x3=True, is_first=False, block_name=''):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        if downsample_in3x3 == True:
            stride1, stride2 = 1, stride
        else:
            stride1, stride2 = stride, 1
        self.is_first = is_first

        self.conv1 = Conv2dUnit(in_c,     filters1, 1, stride=stride1, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', name=block_name+'.conv1')
        self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=stride2, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', use_dcn=use_dcn, name=block_name+'.conv2')
        self.conv3 = Conv2dUnit(filters2, filters3, 1, stride=1, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'.conv3')

        if not self.is_first:
            self.conv4 = Conv2dUnit(in_c, filters3, 1, stride=1, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'.conv4')
        else:
            self.conv4 = Conv2dUnit(in_c, filters3, 1, stride=stride, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'.conv4')

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        if not self.is_first:
            input_tensor = L.pool2d(
                input=input_tensor,
                pool_size=2,
                pool_stride=2,
                pool_padding=0,
                pool_type='avg')
        shortcut = self.conv4(input_tensor)
        x = x + shortcut
        x = L.relu(x)
        return x


class IdentityBlock(object):
    def __init__(self, in_c, filters, norm_type, freeze_norm, norm_decay, lr, use_dcn=False, block_name=''):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = Conv2dUnit(in_c,     filters1, 1, stride=1, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', name=block_name+'.conv1')
        self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=1, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', use_dcn=use_dcn, name=block_name+'.conv2')
        self.conv3 = Conv2dUnit(filters2, filters3, 1, stride=1, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'.conv3')

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + input_tensor
        x = L.relu(x)
        return x

class Resnet50Vd(object):
    def __init__(self, norm_type='bn',
                 feature_maps=[3, 4, 5],
                 dcn_v2_stages=[5],
                 downsample_in3x3=True,
                 freeze_at=0,
                 fix_bn_mean_var_at=0,
                 freeze_norm=False,
                 norm_decay=0.,
                 lr_mult_list=[1., 1., 1., 1.],
                 name='backbone'):
        super(Resnet50Vd, self).__init__()
        self.norm_type = norm_type
        self.feature_maps = feature_maps
        assert freeze_at in [0, 1, 2, 3, 4, 5]
        assert fix_bn_mean_var_at in [0, 1, 2, 3, 4, 5]
        assert len(lr_mult_list) == 4, "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        self.lr_mult_list = lr_mult_list
        self.freeze_at = freeze_at
        self.fix_bn_mean_var_at = fix_bn_mean_var_at
        self.stage1_conv1_1 = Conv2dUnit(3,  32, 3, stride=2, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name=name+'.stage1_conv1_1')
        self.stage1_conv1_2 = Conv2dUnit(32, 32, 3, stride=1, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name=name+'.stage1_conv1_2')
        self.stage1_conv1_3 = Conv2dUnit(32, 64, 3, stride=1, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name=name+'.stage1_conv1_3')

        # stage2
        self.stage2_0 = ConvBlock(64, [64, 64, 256], norm_type, freeze_norm, norm_decay, lr_mult_list[0], stride=1, downsample_in3x3=downsample_in3x3, is_first=True, block_name=name+'.stage2_0')
        self.stage2_1 = IdentityBlock(256, [64, 64, 256], norm_type, freeze_norm, norm_decay, lr_mult_list[0], block_name=name+'.stage2_1')
        self.stage2_2 = IdentityBlock(256, [64, 64, 256], norm_type, freeze_norm, norm_decay, lr_mult_list[0], block_name=name+'.stage2_2')

        # stage3
        use_dcn = 3 in dcn_v2_stages
        self.stage3_0 = ConvBlock(256, [128, 128, 512], norm_type, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name=name+'.stage3_0')
        self.stage3_1 = IdentityBlock(512, [128, 128, 512], norm_type, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name=name+'.stage3_1')
        self.stage3_2 = IdentityBlock(512, [128, 128, 512], norm_type, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name=name+'.stage3_2')
        self.stage3_3 = IdentityBlock(512, [128, 128, 512], norm_type, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name=name+'.stage3_3')

        # stage4
        use_dcn = 4 in dcn_v2_stages
        self.stage4_0 = ConvBlock(512, [256, 256, 1024], norm_type, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name=name+'.stage4_0')
        self.stage4_1 = IdentityBlock(1024, [256, 256, 1024], norm_type, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name=name+'.stage4_1')
        self.stage4_2 = IdentityBlock(1024, [256, 256, 1024], norm_type, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name=name+'.stage4_2')
        self.stage4_3 = IdentityBlock(1024, [256, 256, 1024], norm_type, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name=name+'.stage4_3')
        self.stage4_4 = IdentityBlock(1024, [256, 256, 1024], norm_type, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name=name+'.stage4_4')
        self.stage4_5 = IdentityBlock(1024, [256, 256, 1024], norm_type, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name=name+'.stage4_5')

        # stage5
        use_dcn = 5 in dcn_v2_stages
        self.stage5_0 = ConvBlock(1024, [512, 512, 2048], norm_type, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name=name+'.stage5_0')
        self.stage5_1 = IdentityBlock(2048, [512, 512, 2048], norm_type, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, block_name=name+'.stage5_1')
        self.stage5_2 = IdentityBlock(2048, [512, 512, 2048], norm_type, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, block_name=name+'.stage5_2')

    def __call__(self, input_tensor):
        x = self.stage1_conv1_1(input_tensor)
        x = self.stage1_conv1_2(x)
        x = self.stage1_conv1_3(x)
        x = L.pool2d(
            input=x,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        # stage2
        x = self.stage2_0(x)
        x = self.stage2_1(x)
        s4 = self.stage2_2(x)
        # stage3
        x = self.stage3_0(s4)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        s8 = self.stage3_3(x)
        # stage4
        x = self.stage4_0(s8)
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        x = self.stage4_3(x)
        x = self.stage4_4(x)
        s16 = self.stage4_5(x)
        # stage5
        x = self.stage5_0(s16)
        x = self.stage5_1(x)
        s32 = self.stage5_2(x)

        outs = []
        if 2 in self.feature_maps:
            outs.append(s4)
        if 3 in self.feature_maps:
            outs.append(s8)
        if 4 in self.feature_maps:
            outs.append(s16)
        if 5 in self.feature_maps:
            outs.append(s32)
        return outs



class BasicBlock(object):
    def __init__(self, in_c, filters, norm_type, freeze_norm, norm_decay, lr, stride=1, is_first=False, block_name=''):
        super(BasicBlock, self).__init__()
        filters1, filters2 = filters
        stride1, stride2 = stride, 1
        self.is_first = is_first
        self.stride = stride

        self.conv1 = Conv2dUnit(in_c,     filters1, 3, stride=stride1, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', name=block_name+'_branch2a')
        self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=stride2, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'_branch2b')

        self.conv3 = None
        if self.stride == 2 or self.is_first:
            if not self.is_first:
                self.conv3 = Conv2dUnit(in_c, filters2, 1, stride=1, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'_branch1')
            else:
                self.conv3 = Conv2dUnit(in_c, filters2, 1, stride=stride, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'_branch1')

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        if self.stride == 2 or self.is_first:
            if not self.is_first:
                input_tensor = L.pool2d(
                    input=input_tensor,
                    pool_size=2,
                    pool_stride=2,
                    pool_padding=0,
                    pool_type='avg')
            shortcut = self.conv3(input_tensor)
        else:
            shortcut = input_tensor
        x = x + shortcut
        x = L.relu(x)
        return x


class Resnet18Vd(object):
    def __init__(self, norm_type='bn',
                 feature_maps=[4, 5],
                 dcn_v2_stages=[],
                 freeze_at=0,
                 fix_bn_mean_var_at=0,
                 freeze_norm=False,
                 norm_decay=0.,
                 lr_mult_list=[1., 1., 1., 1.]):
        super(Resnet18Vd, self).__init__()
        self.norm_type = norm_type
        self.feature_maps = feature_maps
        assert freeze_at in [0, 1, 2, 3, 4, 5]
        assert fix_bn_mean_var_at in [0, 1, 2, 3, 4, 5]
        assert len(lr_mult_list) == 4, "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        self.lr_mult_list = lr_mult_list
        self.freeze_at = freeze_at
        self.fix_bn_mean_var_at = fix_bn_mean_var_at
        self.stage1_conv1_1 = Conv2dUnit(3,  32, 3, stride=2, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_1')
        self.stage1_conv1_2 = Conv2dUnit(32, 32, 3, stride=1, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_2')
        self.stage1_conv1_3 = Conv2dUnit(32, 64, 3, stride=1, norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_3')

        # stage2
        self.stage2_0 = BasicBlock(64, [64, 64], norm_type, freeze_norm, norm_decay, lr_mult_list[0], stride=1, is_first=True, block_name='res2a')
        self.stage2_1 = BasicBlock(64, [64, 64], norm_type, freeze_norm, norm_decay, lr_mult_list[0], stride=1, block_name='res2b')

        # stage3
        self.stage3_0 = BasicBlock(64, [128, 128], norm_type, freeze_norm, norm_decay, lr_mult_list[1], stride=2, block_name='res3a')
        self.stage3_1 = BasicBlock(128, [128, 128], norm_type, freeze_norm, norm_decay, lr_mult_list[1], stride=1, block_name='res3b')

        # stage4
        self.stage4_0 = BasicBlock(128, [256, 256], norm_type, freeze_norm, norm_decay, lr_mult_list[2], stride=2, block_name='res4a')
        self.stage4_1 = BasicBlock(256, [256, 256], norm_type, freeze_norm, norm_decay, lr_mult_list[2], stride=1, block_name='res4b')

        # stage5
        self.stage5_0 = BasicBlock(256, [512, 512], norm_type, freeze_norm, norm_decay, lr_mult_list[3], stride=2, block_name='res5a')
        self.stage5_1 = BasicBlock(512, [512, 512], norm_type, freeze_norm, norm_decay, lr_mult_list[3], stride=1, block_name='res5b')

    def __call__(self, input_tensor):
        x = self.stage1_conv1_1(input_tensor)
        x = self.stage1_conv1_2(x)
        x = self.stage1_conv1_3(x)
        x = L.pool2d(
            input=x,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        # stage2
        x = self.stage2_0(x)
        s4 = self.stage2_1(x)
        # stage3
        x = self.stage3_0(s4)
        s8 = self.stage3_1(x)
        # stage4
        x = self.stage4_0(s8)
        s16 = self.stage4_1(x)
        # stage5
        x = self.stage5_0(s16)
        s32 = self.stage5_1(x)

        outs = []
        if 2 in self.feature_maps:
            outs.append(s4)
        if 3 in self.feature_maps:
            outs.append(s8)
        if 4 in self.feature_maps:
            outs.append(s16)
        if 5 in self.feature_maps:
            outs.append(s32)
        return outs






