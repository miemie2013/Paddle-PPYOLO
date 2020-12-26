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
import paddle.fluid as fluid
import paddle.fluid.layers as L
import paddle.nn.functional as F
from paddle.fluid import Variable
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import utils
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant, Normal
from paddle.fluid.regularizer import L2Decay




def deformable_conv(input,
                    offset,
                    mask,
                    num_filters,
                    filter_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=None,
                    deformable_groups=None,
                    im2col_step=None,
                    filter_param=None,
                    bias_attr=None,
                    modulated=True,
                    name=None):

    check_variable_and_dtype(input, "input", ['float32', 'float64'],
                             'deformable_conv')
    check_variable_and_dtype(offset, "offset", ['float32', 'float64'],
                             'deformable_conv')
    check_type(mask, 'mask', (Variable, type(None)), 'deformable_conv')

    num_channels = input.shape[1]
    assert filter_param is not None, "filter_param should not be None here."

    helper = LayerHelper('deformable_conv', **locals())
    dtype = helper.input_dtype()

    if not isinstance(input, Variable):
        raise TypeError("Input of deformable_conv must be Variable")
    if not isinstance(offset, Variable):
        raise TypeError("Input Offset of deformable_conv must be Variable")

    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    input_shape = input.shape
    filter_shape = [num_filters, int(num_filter_channels)] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * num_channels
        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    pre_bias = helper.create_variable_for_type_inference(dtype)

    if modulated:
        helper.append_op(
            type='deformable_conv',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
                'Mask': mask,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    else:
        helper.append_op(
            type='deformable_conv_v1',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    output = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    return output



def dcnv2(input,
                    offset,
                    mask,
                    num_filters,
                    filter_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=None,
                    deformable_groups=None,
                    im2col_step=None,
                    filter_param=None,
                    bias_attr=None,
                    modulated=True,
                    name=None):
    x = input
    dcn_weight = filter_param
    N, in_C, H, W = x.shape
    out_C, in_C, kH, kW = dcn_weight.shape
    out_W = (W + 2 * padding - (kW - 1)) // stride
    out_H = (H + 2 * padding - (kH - 1)) // stride

    # 1.先对图片x填充得到填充后的图片pad_x
    pad_x_H = H + padding * 2 + 1
    pad_x_W = W + padding * 2 + 1
    pad_x = L.pad(x, paddings=[0, 0, 0, 0, padding, padding+1, padding, padding+1], pad_value=0.0)

    # 卷积核中心点在pad_x中的位置
    rows = L.range(0., out_W, 1., dtype='float32') * stride + padding     # [out_W, ]
    cols = L.range(0., out_H, 1., dtype='float32') * stride + padding     # [out_H, ]
    rows = L.expand(L.reshape(rows, (1, 1, -1, 1, 1)), [1, out_H, 1, 1, 1])   # [1, out_H, out_W, 1, 1]
    cols = L.expand(L.reshape(cols, (1, -1, 1, 1, 1)), [1, 1, out_W, 1, 1])   # [1, out_H, out_W, 1, 1]
    start_pos_yx = L.concat([cols, rows], -1)   # [1, out_H, out_W, 1, 2]   仅仅是卷积核中心点在pad_x中的位置
    start_pos_yx = L.expand(start_pos_yx, [N, 1, 1, kH * kW, 1])   # [N, out_H, out_W, kH*kW, 2]   仅仅是卷积核中心点在pad_x中的位置
    start_pos_y = start_pos_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
    start_pos_x = start_pos_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置

    # 卷积核内部的偏移
    half_W = (kW - 1) // 2
    half_H = (kW - 1) // 2
    rows2 = L.range(0., kW, 1., dtype='float32') - half_W
    cols2 = L.range(0., kH, 1., dtype='float32') - half_H
    rows2 = L.expand(L.reshape(rows2, (1, -1, 1)), [kH, 1, 1])
    cols2 = L.expand(L.reshape(cols2, (-1, 1, 1)), [1, kW, 1])
    filter_inner_offset_yx = L.concat([cols2, rows2], -1)  # [kH, kW, 2]   卷积核内部的偏移
    filter_inner_offset_yx = L.reshape(filter_inner_offset_yx,
                                       (1, 1, 1, kH * kW, 2))  # [1, 1, 1, kH*kW, 2]   卷积核内部的偏移
    filter_inner_offset_yx = L.expand(filter_inner_offset_yx, [N, out_H, out_W, 1, 1])  # [N, out_H, out_W, kH*kW, 2]   卷积核内部的偏移
    filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
    filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移

    mask = L.transpose(mask, [0, 2, 3, 1])  # [N, out_H, out_W, kH*kW*1]
    offset = L.transpose(offset, [0, 2, 3, 1])  # [N, out_H, out_W, kH*kW*2]
    offset_yx = L.reshape(offset, (N, out_H, out_W, kH * kW, 2))  # [N, out_H, out_W, kH*kW, 2]
    offset_y = offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]
    offset_x = offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]

    # 最终位置。其实也不是最终位置，为了更快速实现DCNv2，还要给y坐标（代表行号）加上图片的偏移来一次性抽取，避免for循环遍历每一张图片。
    start_pos_y.stop_gradient = True
    start_pos_x.stop_gradient = True
    filter_inner_offset_y.stop_gradient = True
    filter_inner_offset_x.stop_gradient = True
    pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
    pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
    pos_y = L.clip(pos_y, 0.0, H + padding * 2 - 1.0)
    pos_x = L.clip(pos_x, 0.0, W + padding * 2 - 1.0)
    ytxt = L.concat([pos_y, pos_x], -1)  # [N, out_H, out_W, kH*kW, 2]

    pad_x = L.transpose(pad_x, [0, 2, 3, 1])  # [N, pad_x_H, pad_x_W, C]
    pad_x = L.reshape(pad_x, (N*pad_x_H, pad_x_W, in_C))  # [N*pad_x_H, pad_x_W, C]

    ytxt = L.reshape(ytxt, (N * out_H * out_W * kH * kW, 2))  # [N*out_H*out_W*kH*kW, 2]
    _yt = ytxt[:, :1]  # [N*out_H*out_W*kH*kW, 1]
    _xt = ytxt[:, 1:]  # [N*out_H*out_W*kH*kW, 1]

    # 为了避免使用for循环遍历每一张图片，还要给y坐标（代表行号）加上图片的偏移来一次性抽取出更兴趣的像素。
    row_offset = L.range(0., N, 1., dtype='float32') * pad_x_H  # [N, ]
    row_offset = L.expand(L.reshape(row_offset, (-1, 1, 1)), [1, out_H * out_W * kH * kW, 1])  # [N, out_H*out_W*kH*kW, 1]
    row_offset = L.reshape(row_offset, (N * out_H * out_W * kH * kW, 1))  # [N*out_H*out_W*kH*kW, 1]
    row_offset.stop_gradient = True
    _yt += row_offset

    _y1 = L.floor(_yt)
    _x1 = L.floor(_xt)
    _y2 = _y1 + 1.0
    _x2 = _x1 + 1.0
    _y1x1 = L.concat([_y1, _x1], -1)
    _y1x2 = L.concat([_y1, _x2], -1)
    _y2x1 = L.concat([_y2, _x1], -1)
    _y2x2 = L.concat([_y2, _x2], -1)

    _y1x1_int = L.cast(_y1x1, 'int32')   # [N*out_H*out_W*kH*kW, 2]
    v1 = L.gather_nd(pad_x, _y1x1_int)   # [N*out_H*out_W*kH*kW, in_C]
    _y1x2_int = L.cast(_y1x2, 'int32')   # [N*out_H*out_W*kH*kW, 2]
    v2 = L.gather_nd(pad_x, _y1x2_int)   # [N*out_H*out_W*kH*kW, in_C]
    _y2x1_int = L.cast(_y2x1, 'int32')   # [N*out_H*out_W*kH*kW, 2]
    v3 = L.gather_nd(pad_x, _y2x1_int)   # [N*out_H*out_W*kH*kW, in_C]
    _y2x2_int = L.cast(_y2x2, 'int32')   # [N*out_H*out_W*kH*kW, 2]
    v4 = L.gather_nd(pad_x, _y2x2_int)   # [N*out_H*out_W*kH*kW, in_C]

    lh = _yt - _y1  # [N*out_H*out_W*kH*kW, 1]
    lw = _xt - _x1
    hh = 1 - lh
    hw = 1 - lw
    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw
    value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4   # [N*out_H*out_W*kH*kW, in_C]
    mask = L.reshape(mask, (N * out_H * out_W * kH * kW, 1))
    value = value * mask
    value = L.reshape(value, (N, out_H, out_W, kH, kW, in_C))
    new_x = L.transpose(value, [0, 1, 2, 5, 3, 4])   # [N, out_H, out_W, in_C, kH, kW]

    # 旧的方案，使用逐元素相乘，慢！
    # new_x = torch.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))  # [N, out_H, out_W, in_C * kH * kW]
    # new_x = new_x.permute(0, 3, 1, 2)  # [N, in_C*kH*kW, out_H, out_W]
    # exp_new_x = new_x.unsqueeze(1)  # 增加1维，[N,      1, in_C*kH*kW, out_H, out_W]
    # reshape_w = torch.reshape(dcn_weight, (1, out_C, in_C * kH * kW, 1, 1))  # [1, out_C,  in_C*kH*kW,     1,     1]
    # out = exp_new_x * reshape_w  # 逐元素相乘，[N, out_C,  in_C*kH*kW, out_H, out_W]
    # out = out.sum((2,))  # 第2维求和，[N, out_C, out_H, out_W]

    # 新的方案，用等价的1x1卷积代替逐元素相乘，快！
    new_x = L.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))  # [N, out_H, out_W, in_C * kH * kW]
    new_x = L.transpose(new_x, [0, 3, 1, 2])  # [N, in_C*kH*kW, out_H, out_W]
    rw = L.reshape(dcn_weight, (out_C, in_C * kH * kW, 1, 1))  # [out_C, in_C, kH, kW] -> [out_C, in_C*kH*kW, 1, 1]  变成1x1卷积核
    out = F.conv2d(new_x, rw, stride=1)  # [N, out_C, out_H, out_W]
    return out



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
    elif norm_type == 'affine_channel':
        af = 1
    return bn, gn, af




class Mish(paddle.nn.Layer):
    def __init__(self):
        super(Mish, self).__init__()

    def _softplus(self, x):
        expf = fluid.layers.exp(fluid.layers.clip(x, -200, 50))
        return fluid.layers.log(1 + expf)

    def __call__(self, x):
        return x * fluid.layers.tanh(self._softplus(x))


class Conv2dUnit(paddle.nn.Layer):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 bias_attr=False,
                 bn=0,
                 gn=0,
                 af=0,
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
        self.groups = groups
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = (filter_size - 1) // 2
        self.act = act
        self.freeze_norm = freeze_norm
        self.is_test = is_test
        self.norm_decay = norm_decay
        self.use_dcn = use_dcn
        self.name = name

        # conv
        conv_name = name
        self.dcn_param = None
        if use_dcn:
            self.conv = paddle.nn.Conv2D(input_dim,
                                         filter_size * filter_size * 3,
                                         kernel_size=filter_size,
                                         stride=stride,
                                         padding=self.padding,
                                         weight_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + "_conv_offset.w_0"),
                                         bias_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + "_conv_offset.b_0"))
            self.dcn_param = fluid.layers.create_parameter(
                shape=[filters, input_dim, filter_size, filter_size],
                dtype='float32',
                attr=ParamAttr(name=conv_name + "_dcn_weights", learning_rate=lr, initializer=weight_init),
                default_initializer=fluid.initializer.Xavier())
        else:
            conv_battr = False
            if bias_attr:
                blr = lr
                if bias_lr:
                    blr = bias_lr
                conv_battr = ParamAttr(name=conv_name + "_bias",
                                       learning_rate=blr,
                                       initializer=bias_init,
                                       regularizer=L2Decay(0.))   # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            self.conv = paddle.nn.Conv2D(input_dim,
                                         filters,
                                         kernel_size=filter_size,
                                         stride=stride,
                                         padding=self.padding,
                                         weight_attr=ParamAttr(name=conv_name + "_weights", learning_rate=lr, initializer=weight_init),
                                         bias_attr=conv_battr)


        # norm
        if conv_name == "conv1":
            norm_name = "bn_" + conv_name
            if gn:
                norm_name = "gn_" + conv_name
            if af:
                norm_name = "af_" + conv_name
        else:
            norm_name = "bn" + conv_name[3:]
            if gn:
                norm_name = "gn" + conv_name[3:]
            if af:
                norm_name = "af" + conv_name[3:]
        norm_lr = 0. if freeze_norm else lr
        pattr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),   # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            name=norm_name + "_scale",
            trainable=False if freeze_norm else True)
        battr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),   # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            name=norm_name + "_offset",
            trainable=False if freeze_norm else True)
        self.bn = None
        self.gn = None
        self.af = None
        if bn:
            self.bn = paddle.nn.BatchNorm2D(filters, weight_attr=pattr, bias_attr=battr)
        if gn:
            self.gn = paddle.nn.GroupNorm(num_groups=groups, num_channels=filters, weight_attr=pattr, bias_attr=battr)
        if af:
            self.af = True
            self.scale = fluid.layers.create_parameter(
                shape=[filters],
                dtype='float32',
                attr=pattr,
                default_initializer=Constant(1.))
            self.offset = fluid.layers.create_parameter(
                shape=[filters],
                dtype='float32',
                attr=battr,
                default_initializer=Constant(0.))

        # act
        self.act = None
        if act == 'relu':
            self.act = paddle.nn.ReLU()
        elif act == 'leaky':
            self.act = paddle.nn.LeakyReLU(0.1)
        elif act == 'mish':
            self.act = Mish()
        elif act is None:
            pass
        else:
            raise NotImplementedError("Activation \'{}\' is not implemented.".format(act))


    def freeze(self):
        if self.conv is not None:
            if self.conv.weight is not None:
                self.conv.weight.stop_gradient = True
            if self.conv.bias is not None:
                self.conv.bias.stop_gradient = True
        if self.dcn_param is not None:
            self.dcn_param.stop_gradient = True
        if self.bn is not None:
            self.bn.weight.stop_gradient = True
            self.bn.bias.stop_gradient = True
        if self.gn is not None:
            self.gn.weight.stop_gradient = True
            self.gn.bias.stop_gradient = True
        if self.af is not None:
            self.scale.stop_gradient = True
            self.offset.stop_gradient = True

    def forward(self, x):
        if self.use_dcn:
            offset_mask = self.conv(x)
            offset = offset_mask[:, :self.filter_size**2 * 2, :, :]
            mask = offset_mask[:, self.filter_size**2 * 2:, :, :]
            mask = fluid.layers.sigmoid(mask)
            x = deformable_conv(input=x, offset=offset, mask=mask,
                                num_filters=self.filters,
                                filter_size=self.filter_size,
                                stride=self.stride,
                                padding=self.padding,
                                groups=1,
                                deformable_groups=1,
                                im2col_step=1,
                                filter_param=self.dcn_param,
                                bias_attr=False)

            # 自实现的DCNv2
            # x = dcnv2(input=x, offset=offset, mask=mask,
            #                     num_filters=self.filters,
            #                     filter_size=self.filter_size,
            #                     stride=self.stride,
            #                     padding=self.padding,
            #                     groups=1,
            #                     deformable_groups=1,
            #                     im2col_step=1,
            #                     filter_param=self.dcn_param,
            #                     bias_attr=False)
        else:
            x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.gn:
            x = self.gn(x)
        if self.af:
            x = fluid.layers.affine_channel(x, scale=self.scale, bias=self.offset, act=None)
        if self.act:
            x = self.act(x)
        return x


class CoordConv(paddle.nn.Layer):
    def __init__(self, coord_conv=True):
        super(CoordConv, self).__init__()
        self.coord_conv = coord_conv

    def __call__(self, input):
        if not self.coord_conv:
            return input
        b = input.shape[0]
        h = input.shape[2]
        w = input.shape[3]
        x_range = L.range(0, w, 1., dtype='float32') / (w - 1) * 2.0 - 1
        y_range = L.range(0, h, 1., dtype='float32') / (h - 1) * 2.0 - 1
        # x_range = paddle.to_tensor(x_range, place=input.place)
        # y_range = paddle.to_tensor(y_range, place=input.place)
        x_range = L.reshape(x_range, (1, 1, 1, -1))  # [1, 1, 1, w]
        y_range = L.reshape(y_range, (1, 1, -1, 1))  # [1, 1, h, 1]
        x_range = L.expand(x_range, [b, 1, h, 1])  # [b, 1, h, w]
        y_range = L.expand(y_range, [b, 1, 1, w])  # [b, 1, h, w]
        offset = L.concat([input, x_range, y_range], axis=1)
        return offset


class SPP(paddle.nn.Layer):
    def __init__(self, seq='asc'):
        super(SPP, self).__init__()
        assert seq in ['desc', 'asc']
        self.seq = seq
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=5, stride=1, padding=2)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=9, stride=1, padding=4)
        self.max_pool3 = paddle.nn.MaxPool2D(kernel_size=13, stride=1, padding=6)

    def __call__(self, x):
        x_1 = x
        x_2 = self.max_pool1(x)
        x_3 = self.max_pool2(x)
        x_4 = self.max_pool3(x)
        if self.seq == 'desc':
            out = L.concat([x_4, x_3, x_2, x_1], axis=1)
        else:
            out = L.concat([x_1, x_2, x_3, x_4], axis=1)
        return out


class DropBlock(paddle.nn.Layer):
    def __init__(self,
                 block_size=3,
                 keep_prob=0.9,
                 is_test=False):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.is_test = is_test

    def __call__(self, input):
        if self.is_test:
            return input

        def CalculateGamma(input, block_size, keep_prob):
            input_shape = fluid.layers.shape(input)
            feat_shape_tmp = fluid.layers.slice(input_shape, [0], [3], [4])
            feat_shape_tmp = fluid.layers.cast(feat_shape_tmp, dtype="float32")
            feat_shape_t = fluid.layers.reshape(feat_shape_tmp, [1, 1, 1, 1])
            feat_area = fluid.layers.pow(feat_shape_t, factor=2)

            block_shape_t = fluid.layers.fill_constant(
                shape=[1, 1, 1, 1], value=block_size, dtype='float32')
            block_area = fluid.layers.pow(block_shape_t, factor=2)

            useful_shape_t = feat_shape_t - block_shape_t + 1
            useful_area = fluid.layers.pow(useful_shape_t, factor=2)

            upper_t = feat_area * (1 - keep_prob)
            bottom_t = block_area * useful_area
            output = upper_t / bottom_t
            return output

        gamma = CalculateGamma(input, block_size=self.block_size, keep_prob=self.keep_prob)
        input_shape = fluid.layers.shape(input)
        p = fluid.layers.expand_as(gamma, input)

        input_shape_tmp = fluid.layers.cast(input_shape, dtype="int64")
        random_matrix = fluid.layers.uniform_random(
            input_shape_tmp, dtype='float32', min=0.0, max=1.0)
        one_zero_m = fluid.layers.less_than(random_matrix, p)
        one_zero_m.stop_gradient = True
        one_zero_m = fluid.layers.cast(one_zero_m, dtype="float32")

        mask_flag = fluid.layers.pool2d(
            one_zero_m,
            pool_size=self.block_size,
            pool_type='max',
            pool_stride=1,
            pool_padding=self.block_size // 2)
        mask = 1.0 - mask_flag

        elem_numel = fluid.layers.reduce_prod(input_shape)
        elem_numel_m = fluid.layers.cast(elem_numel, dtype="float32")
        elem_numel_m.stop_gradient = True

        elem_sum = fluid.layers.reduce_sum(mask)
        elem_sum_m = fluid.layers.cast(elem_sum, dtype="float32")
        elem_sum_m.stop_gradient = True

        output = input * mask * elem_numel_m / elem_sum_m
        return output



