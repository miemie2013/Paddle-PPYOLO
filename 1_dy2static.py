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
from static_model.ppyolo import PPYOLO
import static_model.get_model as M
from collections import OrderedDict
import paddle.fluid as fluid
import paddle
import os

print(paddle.__version__)
paddle.disable_static()



cfg = PPYOLO_2x_Config()
# 该模型是COCO数据集上训练好的，所以强制改类别数为80
cfg.num_classes = 80
cfg.head['num_classes'] = cfg.num_classes




model_path = 'dygraph_ppyolo_2x.pdparams'
save_name = 'static_ppyolo_2x'





state_dict = paddle.load(model_path)
print('============================================================')
paddle.enable_static()



# 创建模型
startup_prog = fluid.Program()
infer_prog = fluid.Program()
with fluid.program_guard(infer_prog, startup_prog):
    with fluid.unique_name.guard():
        # 创建模型
        Backbone = M.select_backbone(cfg.backbone_type)
        backbone = Backbone(**cfg.backbone)
        Head = M.select_head(cfg.head_type)
        head = Head(yolo_loss=None, nms_cfg=cfg.nms_cfg, **cfg.head)
        model = PPYOLO(backbone, head)

        image = L.data(name='image', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
        im_size = L.data(name='im_size', shape=[-1, 2], append_batch_size=False, dtype='int32')
        feed_vars = [('image', image), ('im_size', im_size)]
        feed_vars = OrderedDict(feed_vars)
        pred = model(image, im_size)
        test_fetches = {'pred': pred, }
infer_prog = infer_prog.clone(for_test=True)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)


print('\nCopying...')


def copy(name, w):
    print(name)
    tensor = fluid.global_scope().find_var(name).get_tensor()
    tensor.set(w, place)


for key in state_dict.keys():
    copy(key, state_dict[key])


fluid.save(infer_prog, save_name)
# fluid.io.save_persistables(exe, save_name, infer_prog)

if os.path.exists(save_name+'.pdopt'): os.remove(save_name+'.pdopt')
if os.path.exists(save_name+'.pdmodel'): os.remove(save_name+'.pdmodel')
print('\nDone.')






