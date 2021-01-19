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
from numbers import Integral


class ConvBlock(paddle.nn.Layer):
    def __init__(self, in_c, filters, bn, gn, af, freeze_norm, norm_decay, lr, use_dcn=False, stride=2, downsample_in3x3=True, block_name=''):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        if downsample_in3x3 == True:
            stride1, stride2 = 1, stride
        else:
            stride1, stride2 = stride, 1

        self.conv1 = Conv2dUnit(in_c,     filters1, 1, stride=stride1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', name=block_name+'_branch2a')
        self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=stride2, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', use_dcn=use_dcn, name=block_name+'_branch2b')
        self.conv3 = Conv2dUnit(filters2, filters3, 1, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'_branch2c')

        self.conv4 = Conv2dUnit(in_c, filters3, 1, stride=stride, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'_branch1')
        self.act = paddle.nn.ReLU()

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()
        self.conv3.freeze()
        self.conv4.freeze()

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        shortcut = self.conv4(input_tensor)
        x = x + shortcut
        x = self.act(x)
        return x



class HardSwish(paddle.nn.Layer):
    def __init__(self):
        super(HardSwish, self).__init__()

    def __call__(self, x):
        return x * fluid.layers.relu6(x + 3) / 6.


class ConvBNLayer(paddle.nn.Layer):
    def __init__(self,
                 input_dim,
                 num_filters,
                 filter_size,
                 stride,
                 padding,
                 lr_mult=1.0,
                 conv_decay=0.0,
                 norm_type='bn',
                 freeze_norm=False,
                 norm_decay=0.0,
                 num_groups=1,
                 if_act=True,
                 act=None,
                 name=None,
                 use_cudnn=True):
        super(ConvBNLayer, self).__init__()
        self.lr_mult = lr_mult
        self.filter_size = filter_size
        self.stride = stride
        self.act = act
        self.name = name

        # conv
        conv_name = name
        self.conv = paddle.nn.Conv2D(input_dim,
                                     num_filters,
                                     kernel_size=filter_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=num_groups,
                                     weight_attr=ParamAttr(name=conv_name + "_weights",
                                                           learning_rate=lr_mult,
                                                           regularizer=L2Decay(conv_decay)),
                                     bias_attr=False)


        # norm
        norm_name = name + '_bn'
        norm_lr = 0. if freeze_norm else lr_mult
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
        self.af = None
        assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel']
        bn, gn, af = get_norm(norm_type)
        if bn:
            self.bn = paddle.nn.BatchNorm2D(num_filters, weight_attr=pattr, bias_attr=battr)
        if af:
            self.af = True
            self.scale = fluid.layers.create_parameter(
                shape=[num_filters],
                dtype='float32',
                attr=pattr,
                default_initializer=Constant(1.))
            self.offset = fluid.layers.create_parameter(
                shape=[num_filters],
                dtype='float32',
                attr=battr,
                default_initializer=Constant(0.))

        # act
        self.act = None
        if if_act:
            if act == 'relu':
                self.act = paddle.nn.ReLU()
            elif act == 'hard_swish':
                self.act = HardSwish()
            elif act == 'relu6':
                self.act = paddle.nn.ReLU6()


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
        if self.af is not None:
            self.scale.stop_gradient = True
            self.offset.stop_gradient = True

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.af:
            x = fluid.layers.affine_channel(x, scale=self.scale, bias=self.offset, act=None)
        if self.act:
            x = self.act(x)
        return x




class SeBlock(paddle.nn.Layer):
    def __init__(self,
                 in_c,
                 num_out_filter,
                 ratio=4,
                 name=None,

                 lr_mult=1.0,
                 conv_decay=0.0):
        super(SeBlock, self).__init__()
        self.lr_mult = lr_mult
        self.num_out_filter = num_out_filter

        num_mid_filter = int(num_out_filter // ratio)
        self.conv1 = paddle.nn.Conv2D(in_c, num_mid_filter, kernel_size=1, stride=1, padding=0, groups=1,
                                      weight_attr=ParamAttr(name=name + "_1_weights",
                                                            learning_rate=lr_mult,
                                                            regularizer=L2Decay(conv_decay)),
                                      bias_attr=ParamAttr(name=name + "_1_offset",
                                                          learning_rate=lr_mult,
                                                          regularizer=L2Decay(conv_decay)))
        self.relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(num_mid_filter, num_out_filter, kernel_size=1, stride=1, padding=0, groups=1,
                                      weight_attr=ParamAttr(name=name + "_2_weights",
                                                            learning_rate=lr_mult,
                                                            regularizer=L2Decay(conv_decay)),
                                      bias_attr=ParamAttr(name=name + "_2_offset",
                                                          learning_rate=lr_mult,
                                                          regularizer=L2Decay(conv_decay)))
        self.hard_sigmoid = paddle.nn.Hardsigmoid()


    def freeze(self):
        pass

    def forward(self, x):
        pool = fluid.layers.pool2d(x, pool_type='avg', global_pooling=True, use_cudnn=False)
        conv1 = self.conv1(pool)
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.hard_sigmoid(conv2)
        scale = fluid.layers.elementwise_mul(x, y=conv2, axis=0)
        return scale



class ResidualUnit(paddle.nn.Layer):
    def __init__(self,
                 in_c,
                 num_in_filter,
                 num_mid_filter,
                 num_out_filter,
                 stride,
                 filter_size,
                 act=None,
                 use_se=False,
                 name=None,

                 lr_mult=1.0,
                 conv_decay=0.0,
                 norm_type='bn',
                 freeze_norm=False,
                 norm_decay=0.0):
        super(ResidualUnit, self).__init__()
        self.lr_mult = lr_mult
        self.filter_size = filter_size
        self.num_in_filter = num_in_filter
        self.num_out_filter = num_out_filter
        self.stride = stride
        self.use_se = use_se

        self.expand = ConvBNLayer(in_c, num_mid_filter, 1, stride=1, padding=0, lr_mult=lr_mult, conv_decay=conv_decay,
                                  norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay,
                                  num_groups=1, if_act=True, act=act, name=name + '_expand')

        self.res_conv1_depthwise = ConvBNLayer(num_mid_filter, num_mid_filter, filter_size, stride=stride, padding=int((filter_size - 1) // 2), lr_mult=lr_mult, conv_decay=conv_decay,
                                               norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay,
                                               num_groups=num_mid_filter, if_act=True, act=act, name=name + 'res_conv1._depthwise', use_cudnn=False)

        self.se = None
        if use_se:
            self.se = SeBlock(num_mid_filter, num_mid_filter, name=name + '_se', lr_mult=lr_mult, conv_decay=conv_decay)

        self.linear = ConvBNLayer(num_mid_filter, num_out_filter, 1, stride=1, padding=0, lr_mult=lr_mult, conv_decay=conv_decay,
                                  norm_type=norm_type, freeze_norm=freeze_norm, norm_decay=norm_decay,
                                  num_groups=1, if_act=False, name=name + '_linear')

    def freeze(self):
        pass

    def forward(self, x):
        input_data = x
        conv0 = self.expand(x)
        conv1 = self.res_conv1_depthwise(conv0)
        if self.use_se:
            conv1 = self.se(conv1)
        conv2 = self.linear(conv1)

        if self.num_in_filter != self.num_out_filter or self.stride != 1:
            pass
        else:
            conv2 = fluid.layers.elementwise_add(x=input_data, y=conv2, act=None)
        return conv0, conv2







class MobileNetV3(paddle.nn.Layer):
    def __init__(
            self,
            scale=1.0,
            model_name='small',
            feature_maps=[5, 6, 7, 8, 9, 10],
            conv_decay=0.0,
            norm_type='bn',
            norm_decay=0.0,
            extra_block_filters=[[256, 512], [128, 256], [128, 256], [64, 128]],
            lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
            freeze_norm=False,
            multiplier=1.0):
        super(MobileNetV3, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel']
        bn, gn, af = get_norm(norm_type)
        self.bn = bn
        self.gn = gn
        self.af = af
        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]

        if norm_type == 'sync_bn' and freeze_norm:
            raise ValueError(
                "The norm_type should not be sync_bn when freeze_norm is True")
        self.scale = scale
        self.model_name = model_name
        self.feature_maps = feature_maps
        self.extra_block_filters = extra_block_filters
        self.conv_decay = conv_decay
        self.norm_decay = norm_decay
        self.inplanes = 16
        self.end_points = []
        self.block_stride = 0

        self.lr_mult_list = lr_mult_list
        self.freeze_norm = freeze_norm
        self.norm_type = norm_type
        self.curr_stage = 0

        if model_name == "large":
            self.cfg = [
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
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
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
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError

        if multiplier != 1.0:
            self.cfg[-3][2] = int(self.cfg[-3][2] * multiplier)
            self.cfg[-2][1] = int(self.cfg[-2][1] * multiplier)
            self.cfg[-2][2] = int(self.cfg[-2][2] * multiplier)
            self.cfg[-1][1] = int(self.cfg[-1][1] * multiplier)
            self.cfg[-1][2] = int(self.cfg[-1][2] * multiplier)
        self._init_layers()

    def _make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _init_layers(self):
        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        blocks = []

        num_filters = self._make_divisible(inplanes * scale)
        lr_idx = self.curr_stage // 3
        lr_idx = min(lr_idx, len(self.lr_mult_list) - 1)
        lr_mult = self.lr_mult_list[lr_idx]
        self.conv1 = ConvBNLayer(3, num_filters, 3, stride=2, padding=1, lr_mult=lr_mult, conv_decay=self.conv_decay,
                                 norm_type=self.norm_type, freeze_norm=self.freeze_norm, norm_decay=self.norm_decay,
                                 num_groups=1, if_act=True, act='hard_swish', name='conv1')
        i = 0
        inplanes = self._make_divisible(inplanes * scale)
        self._residual_units = paddle.nn.LayerList()
        in_c = num_filters
        for layer_cfg in cfg:
            if layer_cfg[5] == 2:
                self.block_stride += 1

            residual_unit = ResidualUnit(
                in_c,
                num_in_filter=inplanes,
                num_mid_filter=self._make_divisible(scale * layer_cfg[1]),
                num_out_filter=self._make_divisible(scale * layer_cfg[2]),
                act=layer_cfg[4],
                stride=layer_cfg[5],
                filter_size=layer_cfg[0],
                use_se=layer_cfg[3],
                name='conv' + str(i + 2),
                lr_mult=lr_mult, conv_decay=self.conv_decay, norm_type=self.norm_type,
                freeze_norm=self.freeze_norm, norm_decay=self.norm_decay)

            if self.block_stride == 4 and layer_cfg[5] == 2:
                self.block_stride += 1

            self._residual_units.append(residual_unit)
            in_c = self._make_divisible(scale * layer_cfg[2])
            inplanes = self._make_divisible(scale * layer_cfg[2])
            i += 1
            self.curr_stage += 1
        self.block_stride += 1

        # extra block
        # check whether conv_extra is needed
        # if self.block_stride < max(self.feature_maps):
        #     pass
        #     i += 1
        # for block_filter in self.extra_block_filters:
        #     conv_extra = self._extra_block_dw(conv_extra, block_filter[0],
        #                                       block_filter[1], 2,
        #                                       'conv' + str(i + 2))
        #     self.block_stride += 1
        #     if self.block_stride in self.feature_maps:
        #         self.end_points.append(conv_extra)
        #     i += 1

    def freeze(self):
        # self.conv1.freeze()
        pass

    def __call__(self, input_tensor):
        end_points = []
        block_stride = 0
        curr_stage = 0


        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg

        conv = self.conv1(input_tensor)

        i = 0
        for layer_cfg in cfg:
            if layer_cfg[5] == 2:
                block_stride += 1
                if block_stride in self.feature_maps:
                    end_points.append(conv)

            conv0, conv = self._residual_units[i](conv)
            if block_stride == 4 and layer_cfg[5] == 2:
                block_stride += 1
                if block_stride in self.feature_maps:
                    end_points.append(conv0)
            i += 1
            curr_stage += 1
        block_stride += 1
        if block_stride in self.feature_maps:
            end_points.append(conv)

        # extra block
        # check whether conv_extra is needed
        # if block_stride < max(self.feature_maps):
        #     pass
        #     i += 1
        # for block_filter in self.extra_block_filters:
        #     conv_extra = self._extra_block_dw(conv_extra, block_filter[0],
        #                                       block_filter[1], 2,
        #                                       'conv' + str(i + 2))
        #     self.block_stride += 1
        #     if self.block_stride in self.feature_maps:
        #         self.end_points.append(conv_extra)
        #     i += 1
        return end_points



