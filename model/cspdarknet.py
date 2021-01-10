#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import paddle
import paddle.nn.functional as F
from model.custom_layers import *


class ResidualBlock(paddle.nn.Layer):
    def __init__(self, input_dim, filters_1, filters_2, norm_type, name=''):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dUnit(input_dim, filters_1, 1, stride=1, bias_attr=False, norm_type=norm_type, act='mish', name=name+'.conv1')
        self.conv2 = Conv2dUnit(filters_1, filters_2, 3, stride=1, bias_attr=False, norm_type=norm_type, act='mish', name=name+'.conv2')

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.conv2(x)
        x = residual + x
        return x

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()

    def fix_bn(self):
        self.conv1.fix_bn()
        self.conv2.fix_bn()


class StackResidualBlock(paddle.nn.Layer):
    def __init__(self, input_dim, filters_1, filters_2, n, norm_type, name=''):
        super(StackResidualBlock, self).__init__()
        self.sequential = paddle.nn.LayerList()
        for i in range(n):
            residual_block = ResidualBlock(input_dim, filters_1, filters_2, norm_type, name=name+'.block%d' % (i,))
            self.sequential.append(residual_block)

    def forward(self, x):
        for residual_block in self.sequential:
            x = residual_block(x)
        return x

    def freeze(self):
        for residual_block in self.sequential:
            residual_block.freeze()

    def fix_bn(self):
        for residual_block in self.sequential:
            residual_block.fix_bn()



class CSPDarknet53(paddle.nn.Layer):
    def __init__(self, norm_type='bn', feature_maps=[3, 4, 5], freeze_at=0, fix_bn_mean_var_at=0, lr_mult_list=[1., 1., 1., 1.]):
        super(CSPDarknet53, self).__init__()
        self.norm_type = norm_type
        self.feature_maps = feature_maps
        assert freeze_at in [0, 1, 2, 3, 4, 5]
        assert fix_bn_mean_var_at in [0, 1, 2, 3, 4, 5]
        # assert len(lr_mult_list) == 4, "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        # self.lr_mult_list = lr_mult_list
        self.freeze_at = freeze_at
        self.fix_bn_mean_var_at = fix_bn_mean_var_at
        self.conv1 = Conv2dUnit(3,  32, 3, stride=1, norm_type=norm_type, act='mish', name='backbone.conv1')

        # stage1
        self.stage1_conv1 = Conv2dUnit(32, 64, 3, stride=2, norm_type=norm_type, act='mish', name='backbone.stage1_conv1')
        self.stage1_conv2 = Conv2dUnit(64, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage1_conv2')
        self.stage1_conv3 = Conv2dUnit(64, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage1_conv3')
        self.stage1_blocks = StackResidualBlock(64, 32, 64, n=1, norm_type=norm_type, name='backbone.stage1_blocks')
        self.stage1_conv4 = Conv2dUnit(64, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage1_conv4')
        self.stage1_conv5 = Conv2dUnit(128, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage1_conv5')

        # stage2
        self.stage2_conv1 = Conv2dUnit(64, 128, 3, stride=2, norm_type=norm_type, act='mish', name='backbone.stage2_conv1')
        self.stage2_conv2 = Conv2dUnit(128, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage2_conv2')
        self.stage2_conv3 = Conv2dUnit(128, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage2_conv3')
        self.stage2_blocks = StackResidualBlock(64, 64, 64, n=2, norm_type=norm_type, name='backbone.stage2_blocks')
        self.stage2_conv4 = Conv2dUnit(64, 64, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage2_conv4')
        self.stage2_conv5 = Conv2dUnit(128, 128, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage2_conv5')

        # stage3
        self.stage3_conv1 = Conv2dUnit(128, 256, 3, stride=2, norm_type=norm_type, act='mish', name='backbone.stage3_conv1')
        self.stage3_conv2 = Conv2dUnit(256, 128, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage3_conv2')
        self.stage3_conv3 = Conv2dUnit(256, 128, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage3_conv3')
        self.stage3_blocks = StackResidualBlock(128, 128, 128, n=8, norm_type=norm_type, name='backbone.stage3_blocks')
        self.stage3_conv4 = Conv2dUnit(128, 128, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage3_conv4')
        self.stage3_conv5 = Conv2dUnit(256, 256, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage3_conv5')

        # stage4
        self.stage4_conv1 = Conv2dUnit(256, 512, 3, stride=2, norm_type=norm_type, act='mish', name='backbone.stage4_conv1')
        self.stage4_conv2 = Conv2dUnit(512, 256, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage4_conv2')
        self.stage4_conv3 = Conv2dUnit(512, 256, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage4_conv3')
        self.stage4_blocks = StackResidualBlock(256, 256, 256, n=8, norm_type=norm_type, name='backbone.stage4_blocks')
        self.stage4_conv4 = Conv2dUnit(256, 256, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage4_conv4')
        self.stage4_conv5 = Conv2dUnit(512, 512, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage4_conv5')

        # stage5
        self.stage5_conv1 = Conv2dUnit(512, 1024, 3, stride=2, norm_type=norm_type, act='mish', name='backbone.stage5_conv1')
        self.stage5_conv2 = Conv2dUnit(1024, 512, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage5_conv2')
        self.stage5_conv3 = Conv2dUnit(1024, 512, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage5_conv3')
        self.stage5_blocks = StackResidualBlock(512, 512, 512, n=4, norm_type=norm_type, name='backbone.stage5_blocks')
        self.stage5_conv4 = Conv2dUnit(512, 512, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage5_conv4')
        self.stage5_conv5 = Conv2dUnit(1024, 1024, 1, stride=1, norm_type=norm_type, act='mish', name='backbone.stage5_conv5')

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)

        # stage1
        x = self.stage1_conv1(x)
        s2 = self.stage1_conv2(x)
        x = self.stage1_conv3(x)
        x = self.stage1_blocks(x)
        x = self.stage1_conv4(x)
        x = L.concat([x, s2], 1)
        s2 = self.stage1_conv5(x)
        # stage2
        x = self.stage2_conv1(s2)
        s4 = self.stage2_conv2(x)
        x = self.stage2_conv3(x)
        x = self.stage2_blocks(x)
        x = self.stage2_conv4(x)
        x = L.concat([x, s4], 1)
        s4 = self.stage2_conv5(x)
        # stage3
        x = self.stage3_conv1(s4)
        s8 = self.stage3_conv2(x)
        x = self.stage3_conv3(x)
        x = self.stage3_blocks(x)
        x = self.stage3_conv4(x)
        x = L.concat([x, s8], 1)
        s8 = self.stage3_conv5(x)
        # stage4
        x = self.stage4_conv1(s8)
        s16 = self.stage4_conv2(x)
        x = self.stage4_conv3(x)
        x = self.stage4_blocks(x)
        x = self.stage4_conv4(x)
        x = L.concat([x, s16], 1)
        s16 = self.stage4_conv5(x)
        # stage5
        x = self.stage5_conv1(s16)
        s32 = self.stage5_conv2(x)
        x = self.stage5_conv3(x)
        x = self.stage5_blocks(x)
        x = self.stage5_conv4(x)
        x = L.concat([x, s32], 1)
        s32 = self.stage5_conv5(x)

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

    def get_block(self, name):
        layer = getattr(self, name)
        return layer

    def freeze(self):
        freeze_at = self.freeze_at
        if freeze_at >= 1:
            self.conv1.freeze()
            self.stage1_conv1.freeze()
            self.stage1_conv2.freeze()
            self.stage1_conv3.freeze()
            self.stage1_blocks.freeze()
            self.stage1_conv4.freeze()
            self.stage1_conv5.freeze()
        if freeze_at >= 2:
            self.stage2_conv1.freeze()
            self.stage2_conv2.freeze()
            self.stage2_conv3.freeze()
            self.stage2_blocks.freeze()
            self.stage2_conv4.freeze()
            self.stage2_conv5.freeze()
        if freeze_at >= 3:
            self.stage3_conv1.freeze()
            self.stage3_conv2.freeze()
            self.stage3_conv3.freeze()
            self.stage3_blocks.freeze()
            self.stage3_conv4.freeze()
            self.stage3_conv5.freeze()
        if freeze_at >= 4:
            self.stage4_conv1.freeze()
            self.stage4_conv2.freeze()
            self.stage4_conv3.freeze()
            self.stage4_blocks.freeze()
            self.stage4_conv4.freeze()
            self.stage4_conv5.freeze()
        if freeze_at >= 5:
            self.stage5_conv1.freeze()
            self.stage5_conv2.freeze()
            self.stage5_conv3.freeze()
            self.stage5_blocks.freeze()
            self.stage5_conv4.freeze()
            self.stage5_conv5.freeze()

    def fix_bn(self):
        fix_bn_mean_var_at = self.fix_bn_mean_var_at
        if fix_bn_mean_var_at >= 1:
            self.conv1.fix_bn()
            self.stage1_conv1.fix_bn()
            self.stage1_conv2.fix_bn()
            self.stage1_conv3.fix_bn()
            self.stage1_blocks.fix_bn()
            self.stage1_conv4.fix_bn()
            self.stage1_conv5.fix_bn()
        if fix_bn_mean_var_at >= 2:
            self.stage2_conv1.fix_bn()
            self.stage2_conv2.fix_bn()
            self.stage2_conv3.fix_bn()
            self.stage2_blocks.fix_bn()
            self.stage2_conv4.fix_bn()
            self.stage2_conv5.fix_bn()
        if fix_bn_mean_var_at >= 3:
            self.stage3_conv1.fix_bn()
            self.stage3_conv2.fix_bn()
            self.stage3_conv3.fix_bn()
            self.stage3_blocks.fix_bn()
            self.stage3_conv4.fix_bn()
            self.stage3_conv5.fix_bn()
        if fix_bn_mean_var_at >= 4:
            self.stage4_conv1.fix_bn()
            self.stage4_conv2.fix_bn()
            self.stage4_conv3.fix_bn()
            self.stage4_blocks.fix_bn()
            self.stage4_conv4.fix_bn()
            self.stage4_conv5.fix_bn()
        if fix_bn_mean_var_at >= 5:
            self.stage5_conv1.fix_bn()
            self.stage5_conv2.fix_bn()
            self.stage5_conv3.fix_bn()
            self.stage5_blocks.fix_bn()
            self.stage5_conv4.fix_bn()
            self.stage5_conv5.fix_bn()





