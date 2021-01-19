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
from paddle import nn
import paddle.fluid.layers as L
import paddle.nn.functional as F
import math

from model.custom_layers import *



class BasicBlock(paddle.nn.Layer):
    def __init__(self, norm_type, inplanes, planes, stride=1, name=''):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dUnit(inplanes, planes, 3, stride=stride, bias_attr=False, norm_type=norm_type, act='relu', name=name+'.conv1')
        self.conv2 = Conv2dUnit(planes, planes, 3, stride=1, bias_attr=False, norm_type=norm_type, act=None, name=name+'.conv2')
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()

    def fix_bn(self):
        self.conv1.fix_bn()
        self.conv2.fix_bn()



class Root(paddle.nn.Layer):
    def __init__(self, norm_type, in_channels, out_channels, kernel_size, residual, name=''):
        super(Root, self).__init__()
        self.conv = Conv2dUnit(in_channels, out_channels, kernel_size, stride=1, bias_attr=False, norm_type=norm_type, act=None, name=name+'.conv')
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, *x):
        children = x
        x = L.concat(x, 1)
        x = self.conv(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x

    def freeze(self):
        self.conv.freeze()

    def fix_bn(self):
        self.conv.fix_bn()


class Tree(paddle.nn.Layer):
    def __init__(self, norm_type, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1, root_residual=False, name=''):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(norm_type, in_channels, out_channels, stride, name=name+'.tree1')
            self.tree2 = block(norm_type, out_channels, out_channels, 1, name=name+'.tree2')
        else:
            self.tree1 = Tree(norm_type, levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size, root_residual=root_residual, name=name+'.tree1')
            self.tree2 = Tree(norm_type, levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size, root_residual=root_residual, name=name+'.tree2')
        if levels == 1:
            self.root = Root(norm_type, root_dim, out_channels, root_kernel_size, root_residual, name=name+'.root')
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2D(kernel_size=stride, stride=stride, padding=0)
        if in_channels != out_channels:
            self.project = Conv2dUnit(in_channels, out_channels, 1, stride=1, bias_attr=False, norm_type=norm_type, act=None, name=name+'.project')

    def forward(self, x, residual=None, children=None):
        if self.training and residual is not None:   # training是父类nn.Layer中的属性。
            x = x + residual.sum() * 0.0
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

    def freeze(self):
        if self.project:
            self.project.freeze()
        self.tree1.freeze()
        if self.levels == 1:
            self.tree2.freeze()
            self.root.freeze()
        else:
            self.tree2.freeze()

    def fix_bn(self):
        if self.project:
            self.project.fix_bn()
        self.tree1.fix_bn()
        if self.levels == 1:
            self.tree2.fix_bn()
            self.root.fix_bn()
        else:
            self.tree2.fix_bn()



class DLA(paddle.nn.Layer):
    def __init__(self, norm_type, levels, channels, block_name='BasicBlock', residual_root=False, feature_maps=[3, 4, 5], freeze_at=0, fix_bn_mean_var_at=0):
        super(DLA, self).__init__()
        self.norm_type = norm_type
        self.channels = channels
        self.feature_maps = feature_maps
        assert freeze_at in [0, 1, 2, 3, 4, 5, 6, 7]
        assert fix_bn_mean_var_at in [0, 1, 2, 3, 4, 5, 6, 7]
        self.freeze_at = freeze_at
        self.fix_bn_mean_var_at = fix_bn_mean_var_at
        block = None
        if block_name == 'BasicBlock':
            block = BasicBlock

        self._out_features = ["level{}".format(i) for i in range(6)]   # 每个特征图的名字
        self._out_feature_channels = {k: channels[i] for i, k in enumerate(self._out_features)}   # 每个特征图的输出通道数
        self._out_feature_strides = {k: 2 ** i for i, k in enumerate(self._out_features)}   # 每个特征图的下采样倍率

        self.base_layer = Conv2dUnit(3, channels[0], 7, stride=1, bias_attr=False, norm_type=norm_type, act='relu', name='dla.base_layer')
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0], name='dla.level0')
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2, name='dla.level1')
        self.level2 = Tree(norm_type, levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root, name='dla.level2')
        self.level3 = Tree(norm_type, levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root, name='dla.level3')
        self.level4 = Tree(norm_type, levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root, name='dla.level4')
        self.level5 = Tree(norm_type, levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root, name='dla.level5')

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))


    def _make_conv_level(self, inplanes, planes, convs, stride=1, name=''):
        modules = []
        for i in range(convs):
            norm_type = self.norm_type
            modules.append(Conv2dUnit(inplanes, planes, 3, stride=stride if i == 0 else 1, bias_attr=False, norm_type=norm_type, act='relu', name=name+'.conv%d'%i))
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        outs = []
        x = self.base_layer(x)
        for i in range(6):
            name = 'level{}'.format(i)
            x = getattr(self, name)(x)
            if i in self.feature_maps:
                outs.append(x)
        return outs

    def freeze(self):
        freeze_at = self.freeze_at
        if freeze_at >= 1:
            self.base_layer.freeze()
        if freeze_at >= 2:
            n = len(self.level0)
            for i in range(n):
                self.level0[i].freeze()
        if freeze_at >= 3:
            n = len(self.level1)
            for i in range(n):
                self.level1[i].freeze()
        if freeze_at >= 4:
            self.level2.freeze()
        if freeze_at >= 5:
            self.level3.freeze()
        if freeze_at >= 6:
            self.level4.freeze()
        if freeze_at >= 7:
            self.level5.freeze()

    def fix_bn(self):
        fix_bn_mean_var_at = self.fix_bn_mean_var_at
        if fix_bn_mean_var_at >= 1:
            self.base_layer.fix_bn()
        if fix_bn_mean_var_at >= 2:
            n = len(self.level0)
            for i in range(n):
                self.level0[i].fix_bn()
        if fix_bn_mean_var_at >= 3:
            n = len(self.level1)
            for i in range(n):
                self.level1[i].fix_bn()
        if fix_bn_mean_var_at >= 4:
            self.level2.fix_bn()
        if fix_bn_mean_var_at >= 5:
            self.level3.fix_bn()
        if fix_bn_mean_var_at >= 6:
            self.level4.fix_bn()
        if fix_bn_mean_var_at >= 7:
            self.level5.fix_bn()






