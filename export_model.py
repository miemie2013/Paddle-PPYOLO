#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-09-05 18:21:37
#   Description : paddle_fcos
#
# ================================================================
import os
import tempfile
import copy
import paddle
import shutil
import argparse
from collections import OrderedDict
import numpy as np
import paddle.fluid as fluid

from config import *
from static_model.ppyolo import PPYOLO
import static_model.get_model as M
from tools.cocotools import get_classes

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='FCOS Export Script')
parser.add_argument('--config', type=int, default=2,
                    choices=[0, 1, 2],
                    help='0 -- fcos_r50_fpn_multiscale_2x.py;  1 -- fcos_rt_r50_fpn_4x.py;  2 -- fcos_rt_dla34_fpn_4x.py.')
args = parser.parse_args()
config_file = args.config


def _strip_postfix(path):
    path, ext = os.path.splitext(path)
    assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'], \
            "Unknown postfix {} from weights".format(ext)
    return path

def _load_state(path):
    if os.path.exists(path + '.pdopt'):
        # XXX another hack to ignore the optimizer state
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + '.pdparams', dst + '.pdparams')
        state = fluid.io.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = fluid.io.load_program_state(path)
    return state

def load_params(exe, prog, path, ignore_params=[]):
    """
    Load model from the given path.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): load weight to which Program object.
        path (string): URL string or loca model path.
        ignore_params (list): ignore variable to load when finetuning.
            It can be specified by finetune_exclude_pretrained_params
            and the usage can refer to docs/advanced_tutorials/TRANSFER_LEARNING.md
    """

    path = _strip_postfix(path)
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    logger.debug('Loading parameters from {}...'.format(path))
    state = _load_state(path)
    fluid.io.set_program_state(prog, state)

def prune_feed_vars(feeded_var_names, target_vars, prog):
    """
    Filter out feed variables which are not in program,
    pruned feed variables are only used in post processing
    on model output, which are not used in program, such
    as im_id to identify image order, im_shape to clip bbox
    in image.
    """
    exist_var_names = []
    prog = prog.clone()
    prog = prog._prune(targets=target_vars)
    global_block = prog.global_block()
    for name in feeded_var_names:
        try:
            v = global_block.var(name)
            exist_var_names.append(str(v.name))
        except Exception:
            logger.info('save_inference_model pruned unused feed '
                        'variables {}'.format(name))
            pass
    return exist_var_names

def save_infer_model(save_dir, exe, feed_vars, test_fetches, infer_prog):
    feed_var_names = [var.name for var in feed_vars.values()]
    fetch_list = sorted(test_fetches.items(), key=lambda i: i[0])
    target_vars = [var[1] for var in fetch_list]
    feed_var_names = prune_feed_vars(feed_var_names, target_vars, infer_prog)
    logger.info("Export inference model to {}, input: {}, output: "
                "{}...".format(save_dir, feed_var_names,
                               [str(var.name) for var in target_vars]))
    fluid.io.save_inference_model(
        save_dir,
        feeded_var_names=feed_var_names,
        target_vars=target_vars,
        executor=exe,
        main_program=infer_prog,
        params_filename="__params__")


def dump_infer_config(save_dir, cfg):
    if os.path.exists('%s/infer_cfg.yml' % save_dir): os.remove('%s/infer_cfg.yml' % save_dir)
    content = ''
    with open('tools/template_cfg.yml', 'r', encoding='utf-8') as f:
        for line in f:
            for key in cfg:
                key2 = '${%s}' % key
                if key2 in line:
                    if key == 'class_names':
                        line = ''
                        for cname in cfg[key]:
                            line += '- %s\n' % cname
                    else:
                        line = line.replace(key2, str(cfg[key]))
                    break
            content += line
    with open('%s/infer_cfg.yml' % save_dir, 'w', encoding='utf-8') as f:
        f.write(content)
        f.close()



if __name__ == '__main__':
    paddle.enable_static()


    # 推理模型保存目录
    save_dir = 'inference_model'

    min_subgraph_size = 3

    # 是否使用Padddle Executor进行推理。
    use_python_inference = False

    # 使用GPU时，默认为fluid, 可选（fluid/trt_fp32/trt_fp16）
    mode = 'fluid'


    # 选择配置
    cfg = PPYOLO_2x_Config()


    # =============== 导出的是demo.py中使用的配置。即test_cfg里的。 ===============
    classes_path = cfg.classes_path

    # 读取的模型
    model_path = cfg.test_cfg['model_path']

    # target_size越大，精度会上升，但速度会下降。
    target_size = cfg.test_cfg['target_size']


    # 对模型输出的预测框再进行一次分数过滤的阈值。设置为0.0表示不再进行分数过滤。
    # 需要修改这个值的话直接编辑导出的inference_model/infer_cfg.yml配置文件，不需要重新导出模型。
    # 总之，inference_model/infer_cfg.yml里的配置可以手动修改，不需要重新导出模型。
    draw_threshold = cfg.test_cfg['draw_thresh']

    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)


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
            im_size = L.data(name='im_size', shape=[-1, 3], append_batch_size=False, dtype='float32')
            feed_vars = [('image', image), ('im_size', im_size)]
            feed_vars = OrderedDict(feed_vars)
            pred = model(image, im_size)
            test_fetches = {'pred': pred, }
    infer_prog = infer_prog.clone(for_test=True)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)


    load_params(exe, infer_prog, model_path)

    save_infer_model(save_dir, exe, feed_vars, test_fetches, infer_prog)

    # 导出配置文件
    cfg2 = {}
    cfg2['arch'] = 'FCOS'
    cfg2['min_subgraph_size'] = min_subgraph_size
    cfg2['use_python_inference'] = use_python_inference
    cfg2['mode'] = mode
    cfg2['draw_threshold'] = draw_threshold


    cfg2['to_rgb'] = cfg.decodeImage['to_rgb']

    cfg2['is_channel_first'] = cfg.normalizeImage['is_channel_first']
    cfg2['is_scale'] = cfg.normalizeImage['is_scale']
    cfg2['mean0'] = cfg.normalizeImage['mean'][0]
    cfg2['mean1'] = cfg.normalizeImage['mean'][1]
    cfg2['mean2'] = cfg.normalizeImage['mean'][2]
    cfg2['std0'] = cfg.normalizeImage['std'][0]
    cfg2['std1'] = cfg.normalizeImage['std'][1]
    cfg2['std2'] = cfg.normalizeImage['std'][2]

    cfg2['target_size'] = target_size
    cfg2['max_size'] = max_size
    cfg2['interp'] = cfg.resizeImage['interp']
    cfg2['use_cv2'] = cfg.resizeImage['use_cv2']

    cfg2['channel_first'] = cfg.permute['channel_first']
    cfg2['to_bgr'] = cfg.permute['to_bgr']

    cfg2['pad_to_stride'] = cfg.padBatch['pad_to_stride']
    cfg2['use_padded_im_info'] = cfg.padBatch['use_padded_im_info']

    cfg2['class_names'] = all_classes

    dump_infer_config(save_dir, cfg2)
    logger.info("Done.")


