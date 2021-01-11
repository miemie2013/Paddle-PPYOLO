#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import paddle.fluid as fluid
import paddle.fluid.layers as L
import paddle.nn.functional as F
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import Constant



def get_norm(norm_type):
    bn = 0
    gn = 0
    af = 0
    if norm_type == 'bn':
        bn = 1
    elif norm_type == 'sync_bn':
        bn = 1
    elif norm_type == 'gn':
        gn = 1
    elif norm_type == 'in':
        gn = 1
    elif norm_type == 'ln':
        gn = 1
    elif norm_type == 'affine_channel':
        af = 1
    return bn, gn, af



def _softplus(input):
    expf = fluid.layers.exp(fluid.layers.clip(input, -200, 50))
    return fluid.layers.log(1 + expf)


def _mish(input):
    return input * fluid.layers.tanh(_softplus(input))


class Conv2dUnit(object):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 bias_attr=False,
                 norm_type=None,
                 groups=32,
                 act=None,
                 freeze_norm=False,
                 is_test=False,
                 norm_decay=0.,
                 lr=1.,
                 bias_lr=None,
                 weight_init=None,
                 bias_init=None,
                 use_dcn=False,
                 name=''):
        super(Conv2dUnit, self).__init__()
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = (self.filter_size - 1) // 2
        self.bias_attr = bias_attr
        self.groups = groups
        self.act = act
        self.freeze_norm = freeze_norm
        self.is_test = is_test
        self.norm_decay = norm_decay
        self.bias_init = bias_init
        self.use_dcn = use_dcn
        self.name = name

        assert norm_type in [None, 'bn', 'sync_bn', 'gn', 'affine_channel', 'in', 'ln']
        self.bn, self.gn, self.af = get_norm(norm_type)
        if norm_type == 'in':
            self.groups = filters
        if norm_type == 'ln':
            self.groups = 1

    def __call__(self, x):
        conv_name = self.name + ".conv"
        if self.use_dcn:
            offset_mask = fluid.layers.conv2d(
                input=x,
                num_filters=self.filter_size * self.filter_size * 3,
                filter_size=self.filter_size,
                stride=self.stride,
                padding=self.padding,
                act=None,
                param_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + ".weight"),
                bias_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + ".bias"),
                name=conv_name + "_conv_offset")
            offset = offset_mask[:, :self.filter_size**2 * 2, :, :]
            mask = offset_mask[:, self.filter_size**2 * 2:, :, :]
            mask = fluid.layers.sigmoid(mask)
            x = fluid.layers.deformable_conv(input=x, offset=offset, mask=mask,
                                             num_filters=self.filters,
                                             filter_size=self.filter_size,
                                             stride=self.stride,
                                             padding=self.padding,
                                             groups=1,
                                             deformable_groups=1,
                                             im2col_step=1,
                                             param_attr=ParamAttr(name=self.name + ".dcn_param"),
                                             bias_attr=False,
                                             name=conv_name + ".conv2d.output.1")
        else:
            battr = None
            if self.bias_attr:
                battr = ParamAttr(name=conv_name + ".bias", initializer=self.bias_init)
            x = fluid.layers.conv2d(
                input=x,
                num_filters=self.filters,
                filter_size=self.filter_size,
                stride=self.stride,
                padding=self.padding,
                act=None,
                param_attr=ParamAttr(name=conv_name + ".weight"),
                bias_attr=battr,
                name=conv_name + '.output.1')
        if self.bn:
            bn_name = self.name + ".bn"
            norm_lr = 0. if self.freeze_norm else 1.   # 归一化层学习率
            norm_decay = self.norm_decay   # 衰减
            pattr = ParamAttr(
                name=bn_name + '.weight',
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))   # L2权重衰减
            battr = ParamAttr(
                name=bn_name + '.bias',
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))   # L2权重衰减
            x = fluid.layers.batch_norm(
                input=x,
                name=bn_name + '.output.1',
                is_test=self.is_test,  # 冻结层时（即trainable=False），bn的均值、标准差也还是会变化，只有设置is_test=True才保证不变
                param_attr=pattr,
                bias_attr=battr,
                moving_mean_name=bn_name + '._mean',
                moving_variance_name=bn_name + '._variance')
        if self.gn:
            gn_name = self.name + ".gn"
            norm_lr = 0. if self.freeze_norm else 1.   # 归一化层学习率
            norm_decay = self.norm_decay   # 衰减
            pattr = ParamAttr(
                name=gn_name + '.weight',
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))   # L2权重衰减
            battr = ParamAttr(
                name=gn_name + '.bias',
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))   # L2权重衰减
            x = fluid.layers.group_norm(
                input=x,
                groups=self.groups,
                name=gn_name + '.output.1',
                param_attr=pattr,
                bias_attr=battr)
        if self.af:
            af_name = self.name + ".af"
            norm_lr = 0. if self.freeze_norm else 1.   # 归一化层学习率
            norm_decay = self.norm_decay   # 衰减
            pattr = ParamAttr(
                name=af_name + '.weight',
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))   # L2权重衰减
            battr = ParamAttr(
                name=af_name + '.bias',
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))   # L2权重衰减
            scale = fluid.layers.create_parameter(
                shape=[x.shape[1]],
                dtype=x.dtype,
                attr=pattr,
                default_initializer=fluid.initializer.Constant(1.))
            bias = fluid.layers.create_parameter(
                shape=[x.shape[1]],
                dtype=x.dtype,
                attr=battr,
                default_initializer=fluid.initializer.Constant(0.))
            x = fluid.layers.affine_channel(x, scale=scale, bias=bias)

        # act
        if self.act == 'relu':
            x = fluid.layers.relu(x)
        elif self.act == 'leaky':
            x = fluid.layers.leaky_relu(x, alpha=0.1)
        elif self.act == 'mish':
            x = _mish(x)
        elif self.act is None:
            pass
        else:
            raise NotImplementedError("Activation \'{}\' is not implemented.".format(self.act))
        return x


class CoordConv(object):
    def __init__(self, coord_conv=True):
        super(CoordConv, self).__init__()
        self.coord_conv = coord_conv

    def __call__(self, input):
        if not self.coord_conv:
            return input
        b = L.shape(input)[0]
        h = L.shape(input)[2]
        w = L.shape(input)[3]
        x_range = L.range(0, w, 1., dtype='float32') / (w - 1) * 2.0 - 1
        y_range = L.range(0, h, 1., dtype='float32') / (h - 1) * 2.0 - 1
        x_range = L.reshape(x_range, (1, 1, 1, -1))  # [1, 1, 1, w]
        y_range = L.reshape(y_range, (1, 1, -1, 1))  # [1, 1, h, 1]
        x_range = L.expand(x_range, [b, 1, h, 1])  # [b, 1, h, w]
        y_range = L.expand(y_range, [b, 1, 1, w])  # [b, 1, h, w]
        offset = L.concat([input, x_range, y_range], axis=1)
        return offset


class SPP(object):
    def __init__(self, seq='asc'):
        super(SPP, self).__init__()
        assert seq in ['desc', 'asc']
        self.seq = seq

    def __call__(self, x):
        x_1 = x
        x_2 = fluid.layers.pool2d(
            input=x,
            pool_size=5,
            pool_type='max',
            pool_stride=1,
            pool_padding=2,
            ceil_mode=True)
        x_3 = fluid.layers.pool2d(
            input=x,
            pool_size=9,
            pool_type='max',
            pool_stride=1,
            pool_padding=4,
            ceil_mode=True)
        x_4 = fluid.layers.pool2d(
            input=x,
            pool_size=13,
            pool_type='max',
            pool_stride=1,
            pool_padding=6,
            ceil_mode=True)
        if self.seq == 'desc':
            out = fluid.layers.concat(input=[x_4, x_3, x_2, x_1], axis=1)
        else:
            out = fluid.layers.concat(input=[x_1, x_2, x_3, x_4], axis=1)
        return out


class DropBlock(object):
    def __init__(self,
                 block_size=3,
                 keep_prob=0.9,
                 is_test=False):
        super(DropBlock, self).__init__()

    def __call__(self, input):
        return input






