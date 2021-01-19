#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-23 09:13:23
#   Description : paddle2.0_ppyolo
#
# ================================================================
from collections import deque
import time
import threading
import datetime
from collections import OrderedDict
import os
import argparse
import textwrap
import json

from config import *
from model.EMA import ExponentialMovingAverage

from tools.argparser import *
from tools.cocotools import get_classes, catid2clsid, clsid2catid
from model.decoders.decode_yolo import *
from model.architectures.yolo import *
from tools.cocotools import eval
from tools.data_process import data_clean, get_samples
from tools.transform import *
from pycocotools.coco import COCO

import logging

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class COCOTrainDataset(paddle.io.Dataset):
    def __init__(self, records, init_iter_id, cfg, sample_transforms, batch_transforms):
        self.records = records
        self.init_iter_id = init_iter_id
        self.cfg = cfg
        self.sample_transforms = sample_transforms
        self.batch_transforms = batch_transforms
        self.num_record = len(records)
        indexes = [i for i in range(self.num_record)]

        max_iters = cfg.train_cfg['max_iters']
        batch_size = cfg.train_cfg['batch_size']
        self.with_mixup = cfg.decodeImage['with_mixup']
        self.with_cutmix = cfg.decodeImage['with_cutmix']
        self.with_mosaic = cfg.decodeImage['with_mosaic']
        mixup_epoch = cfg.train_cfg['mixup_epoch']
        cutmix_epoch = cfg.train_cfg['cutmix_epoch']
        mosaic_epoch = cfg.train_cfg['mosaic_epoch']
        self.context = cfg.context
        self.batch_size = batch_size

        # 一轮的步数。丢弃最后几个样本。
        train_steps = self.num_record // batch_size
        self.mixup_steps = mixup_epoch * train_steps
        self.cutmix_steps = cutmix_epoch * train_steps
        self.mosaic_steps = mosaic_epoch * train_steps

        # 一轮的样本数。丢弃最后几个样本。
        train_samples = train_steps * batch_size

        # 训练样本
        self.indexes = []
        while len(self.indexes) < max_iters * batch_size:
            indexes2 = copy.deepcopy(indexes)
            # 每个epoch之前洗乱
            np.random.shuffle(indexes2)
            indexes2 = indexes2[:train_samples]
            self.indexes += indexes2
        self.indexes = self.indexes[:max_iters * batch_size]

        # 多尺度训练
        sizes = cfg.randomShape['sizes']
        self.shapes = []
        while len(self.shapes) < max_iters:
            shape = np.random.choice(sizes)
            self.shapes.append(shape)

        # 输出几个特征图
        self.n_layers = len(cfg.head['anchor_masks'])

    def __getitem__(self, idx):
        iter_id = idx // self.batch_size
        if iter_id < self.init_iter_id:   # 恢复训练时跳过。
            return np.zeros((1, ), np.float32)

        img_idx = self.indexes[idx]
        shape = self.shapes[iter_id]
        sample = copy.deepcopy(self.records[img_idx])
        sample["curr_iter"] = iter_id

        # 为mixup数据增强做准备
        if self.with_mixup and iter_id <= self.mixup_steps:
            num = len(self.records)
            mix_idx = np.random.randint(0, num)
            while mix_idx == img_idx:   # 为了不选到自己
                mix_idx = np.random.randint(0, num)
            sample['mixup'] = copy.deepcopy(self.records[mix_idx])
            sample['mixup']["curr_iter"] = iter_id

        # 为cutmix数据增强做准备
        if self.with_cutmix and iter_id <= self.cutmix_steps:
            num = len(self.records)
            mix_idx = np.random.randint(0, num)
            while mix_idx == img_idx:   # 为了不选到自己
                mix_idx = np.random.randint(0, num)
            sample['cutmix'] = copy.deepcopy(self.records[mix_idx])
            sample['cutmix']["curr_iter"] = iter_id

        # 为mosaic数据增强做准备
        if self.with_mosaic and iter_id <= self.mosaic_steps:
            num = len(self.records)
            mix_idx = np.random.randint(0, num)
            while mix_idx == img_idx:   # 为了不选到自己
                mix_idx = np.random.randint(0, num)
            sample['mosaic1'] = copy.deepcopy(self.records[mix_idx])
            sample['mosaic1']["curr_iter"] = iter_id

            mix_idx2 = np.random.randint(0, num)
            while mix_idx2 in [img_idx, mix_idx]:   # 为了不重复
                mix_idx2 = np.random.randint(0, num)
            sample['mosaic2'] = copy.deepcopy(self.records[mix_idx2])
            sample['mosaic2']["curr_iter"] = iter_id

            mix_idx3 = np.random.randint(0, num)
            while mix_idx3 in [img_idx, mix_idx, mix_idx2]:   # 为了不重复
                mix_idx3 = np.random.randint(0, num)
            sample['mosaic3'] = copy.deepcopy(self.records[mix_idx3])
            sample['mosaic3']["curr_iter"] = iter_id

        # batch_transforms
        for sample_transform in self.sample_transforms:
            sample = sample_transform(sample, self.context)

        # batch_transforms
        for batch_transform in self.batch_transforms:
            if isinstance(batch_transform, RandomShapeSingle):
                sample = batch_transform(shape, sample, self.context)
            else:
                sample = batch_transform(sample, self.context)

        # 取出感兴趣的项
        image = sample['image'].astype(np.float32)
        gt_bbox = sample['gt_bbox'].astype(np.float32)
        gt_score = sample['gt_score'].astype(np.float32)
        gt_class = sample['gt_class'].astype(np.int32)
        target0 = sample['target0'].astype(np.float32)
        target1 = sample['target1'].astype(np.float32)
        if self.n_layers > 2:
            target2 = sample['target2'].astype(np.float32)
            return image, gt_bbox, gt_score, gt_class, target0, target1, target2
        return image, gt_bbox, gt_score, gt_class, target0, target1

    def __len__(self):
        return len(self.indexes)


def load_weights(model, model_path):
    _state_dict = model.state_dict()
    pretrained_dict = paddle.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if k in _state_dict:
            shape_1 = _state_dict[k].shape
            shape_2 = pretrained_dict[k].shape
            shape_2 = list(shape_2)
            if shape_1 == shape_2:
                new_state_dict[k] = v
            else:
                print('shape mismatch in %s. shape_1=%s, while shape_2=%s.' % (k, shape_1, shape_2))
    _state_dict.update(new_state_dict)
    model.set_state_dict(_state_dict)

def clear_model(save_dir):
    path_dir = os.listdir(save_dir)
    it_ids = []
    for name in path_dir:
        sss = name.split('.')
        if sss[0] == '':
            continue
        if sss[0] == 'best_model':   # 不会删除最优模型
            it_id = 9999999999
        else:
            it_id = int(sss[0])
        it_ids.append(it_id)
    if len(it_ids) >= 11 * 1:
        it_id = min(it_ids)
        pdparams_path = '%s/%d.pdparams' % (save_dir, it_id)
        if os.path.exists(pdparams_path):
            os.remove(pdparams_path)

def calc_lr(iter_id, cfg):
    base_lr = cfg.learningRate['base_lr']
    piecewiseDecay = cfg.learningRate['PiecewiseDecay']
    linearWarmup = cfg.learningRate['LinearWarmup']
    gamma = piecewiseDecay['gamma']
    milestones = piecewiseDecay['milestones']
    start_factor = linearWarmup['start_factor']
    steps = linearWarmup['steps']
    n = len(milestones)
    for i in range(n, 0, -1):
        if iter_id >= milestones[i-1]:
            return base_lr * gamma ** i
    if iter_id <= steps:
        k = (1.0 - start_factor) / steps
        factor = start_factor + k * iter_id
        return base_lr * factor
    return base_lr

def write(filename, logstats):
    with open(filename, 'a', encoding='utf-8') as f:
        f.writelines(logstats + '\n')
        f.close

if __name__ == '__main__':
    parser = ArgParser()
    use_gpu = parser.get_use_gpu()
    cfg = parser.get_cfg()
    print(paddle.__version__)
    paddle.disable_static()   # 开启动态图
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

    # 打印，确认一下使用的配置
    print('\n=============== config message ===============')
    print('config file: %s' % str(type(cfg)))
    if cfg.train_cfg['model_path'] is not None:
        print('pretrained_model: %s' % cfg.train_cfg['model_path'])
    else:
        print('pretrained_model: None')
    print('use_gpu: %s' % str(use_gpu))
    print()

    # 种类id
    _catid2clsid = {}
    _clsid2catid = {}
    _clsid2cname = {}
    with open(cfg.val_path, 'r', encoding='utf-8') as f2:
        dataset_text = ''
        for line in f2:
            line = line.strip()
            dataset_text += line
        eval_dataset = json.loads(dataset_text)
        categories = eval_dataset['categories']
        for clsid, cate_dic in enumerate(categories):
            catid = cate_dic['id']
            cname = cate_dic['name']
            _catid2clsid[catid] = clsid
            _clsid2catid[clsid] = catid
            _clsid2cname[clsid] = cname
    class_names = []
    num_classes = len(_clsid2cname.keys())
    for clsid in range(num_classes):
        class_names.append(_clsid2cname[clsid])


    # 步id，无需设置，会自动读。
    iter_id = 0

    # 创建模型
    Backbone = select_backbone(cfg.backbone_type)
    backbone = Backbone(**cfg.backbone)
    IouLoss = select_loss(cfg.iou_loss_type)
    iou_loss = IouLoss(**cfg.iou_loss)
    iou_aware_loss = None
    if cfg.head['iou_aware']:
        IouAwareLoss = select_loss(cfg.iou_aware_loss_type)
        iou_aware_loss = IouAwareLoss(**cfg.iou_aware_loss)
    Loss = select_loss(cfg.yolo_loss_type)
    yolo_loss = Loss(iou_loss=iou_loss, iou_aware_loss=iou_aware_loss, **cfg.yolo_loss)
    Head = select_head(cfg.head_type)
    head = Head(yolo_loss=yolo_loss, is_train=True, nms_cfg=cfg.nms_cfg, **cfg.head)
    model = PPYOLO(backbone, head)
    _decode = Decode_YOLO(model, class_names, place, cfg, for_test=False)

    # optimizer
    regularization = None
    if cfg.optimizerBuilder['regularizer'] is not None:
        reg_args = cfg.optimizerBuilder['regularizer'].copy()
        reg_type = reg_args['type'] + 'Decay'   # 正则化类型。L1、L2
        reg_factor = reg_args['factor']
        Regularization = select_regularization(reg_type)
        # 在 优化器 中设置正则化。
        # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
        # 如果同时在 可训练参数的ParamAttr 和 优化器optimizer 中设置正则化， 那么在 可训练参数的ParamAttr 中设置的优先级会高于在 optimizer 中的设置。
        # 也就是说，等价于没给    norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数    加正则化。
        regularization = Regularization(reg_factor)
    optim_args = cfg.optimizerBuilder['optimizer'].copy()
    optim_type = optim_args['type']   # 使用哪种优化器。Momentum、Adam、SGD、...之类的。
    Optimizer = select_optimizer(optim_type)
    del optim_args['type']
    optimizer = Optimizer(learning_rate=cfg.learningRate['base_lr'],
                          parameters=model.parameters(),
                          weight_decay=regularization,   # 正则化
                          grad_clip=None,   # 梯度裁剪
                          **optim_args)

    # 加载权重
    if cfg.train_cfg['model_path'] is not None:
        # 加载参数, 跳过形状不匹配的。
        load_weights(model, cfg.train_cfg['model_path'])

        strs = cfg.train_cfg['model_path'].split('weights/')
        if len(strs) == 2:
            iter_id = int(strs[1].split('.')[0])

    # 冻结，使得需要的显存减少。低显存的卡建议这样配置。
    backbone.freeze()
    backbone.fix_bn()

    print('\n=============== Model ===============')
    trainable_params = 0
    nontrainable_params = 0
    for name_, param_ in model.named_parameters():
        mul = np.prod(param_.shape)
        if param_.stop_gradient is False:
            trainable_params += mul
        else:
            nontrainable_params += mul
    total_params = trainable_params + nontrainable_params
    print('Total params: %s' % format(total_params, ","))
    print('Trainable params: %s' % format(trainable_params, ","))
    print('Non-trainable params: %s\n' % format(nontrainable_params, ","))

    ema = None
    if cfg.use_ema:
        ema = ExponentialMovingAverage(model, cfg.ema_decay)
        ema.register()

    # 训练集
    train_dataset = COCO(cfg.train_path)
    train_img_ids = train_dataset.getImgIds()
    train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, cfg.train_pre_path)
    num_train = len(train_records)
    train_indexes = [i for i in range(num_train)]
    # 验证集
    val_dataset = COCO(cfg.val_path)
    val_img_ids = val_dataset.getImgIds()
    val_images = []   # 只跑有gt的图片，跟随PaddleDetection
    for img_id in val_img_ids:
        ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        if len(ins_anno_ids) == 0:
            continue
        img_anno = val_dataset.loadImgs(img_id)[0]
        val_images.append(img_anno)

    batch_size = cfg.train_cfg['batch_size']
    with_mixup = cfg.decodeImage['with_mixup']
    with_cutmix = cfg.decodeImage['with_cutmix']
    with_mosaic = cfg.decodeImage['with_mosaic']
    mixup_epoch = cfg.train_cfg['mixup_epoch']
    cutmix_epoch = cfg.train_cfg['cutmix_epoch']
    mosaic_epoch = cfg.train_cfg['mosaic_epoch']
    context = cfg.context
    # 预处理
    # sample_transforms
    sample_transforms = []
    for preprocess_name in cfg.sample_transforms_seq:
        if preprocess_name == 'decodeImage':
            preprocess = DecodeImage(**cfg.decodeImage)   # 对图片解码。最开始的一步。
        elif preprocess_name == 'mixupImage':
            preprocess = MixupImage(**cfg.mixupImage)      # mixup增强
        elif preprocess_name == 'cutmixImage':
            preprocess = CutmixImage(**cfg.cutmixImage)    # cutmix增强
        elif preprocess_name == 'mosaicImage':
            preprocess = MosaicImage(**cfg.mosaicImage)    # mosaic增强
        elif preprocess_name == 'colorDistort':
            preprocess = ColorDistort(**cfg.colorDistort)  # 颜色扰动
        elif preprocess_name == 'randomExpand':
            preprocess = RandomExpand(**cfg.randomExpand)  # 随机填充
        elif preprocess_name == 'randomCrop':
            preprocess = RandomCrop(**cfg.randomCrop)        # 随机裁剪
        elif preprocess_name == 'randomFlipImage':
            preprocess = RandomFlipImage(**cfg.randomFlipImage)  # 随机翻转
        elif preprocess_name == 'normalizeBox':
            preprocess = NormalizeBox(**cfg.normalizeBox)        # 将物体的左上角坐标、右下角坐标中的横坐标/图片宽、纵坐标/图片高 以归一化坐标。
        elif preprocess_name == 'padBox':
            preprocess = PadBox(**cfg.padBox)         # 如果gt_bboxes的数量少于num_max_boxes，那么填充坐标是0的bboxes以凑够num_max_boxes。
        elif preprocess_name == 'bboxXYXY2XYWH':
            preprocess = BboxXYXY2XYWH(**cfg.bboxXYXY2XYWH)     # sample['gt_bbox']被改写为cx_cy_w_h格式。
        sample_transforms.append(preprocess)
    # batch_transforms
    batch_transforms = []
    for preprocess_name in cfg.batch_transforms_seq:
        if preprocess_name == 'randomShape':
            preprocess = RandomShapeSingle(random_inter=cfg.randomShape['random_inter'])     # 多尺度训练。随机选一个尺度。也随机选一种插值方式。
        elif preprocess_name == 'normalizeImage':
            preprocess = NormalizeImage(**cfg.normalizeImage)     # 图片归一化。先除以255归一化，再减均值除以标准差
        elif preprocess_name == 'permute':
            preprocess = Permute(**cfg.permute)    # 图片从HWC格式变成CHW格式
        elif preprocess_name == 'gt2YoloTarget':
            preprocess = Gt2YoloTargetSingle(**cfg.gt2YoloTarget)   # 填写target张量。
        batch_transforms.append(preprocess)

    print('\n=============== sample_transforms ===============')
    for trf in sample_transforms:
        print('%s' % str(type(trf)))
    print('\n=============== batch_transforms ===============')
    for trf in batch_transforms:
        print('%s' % str(type(trf)))

    train_dataset = COCOTrainDataset(train_records, iter_id, cfg, sample_transforms, batch_transforms)
    # for i in range(len(train_dataset)):
    #     data = train_dataset[i]
    #     print(data)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size,
                                        num_workers=cfg.train_cfg['num_workers'],
                                        use_shared_memory=False,   # use_shared_memory=True且num_workers>0时会报错。
                                        shuffle=False, drop_last=True)

    # 输出几个特征图
    n_layers = len(cfg.head['anchor_masks'])

    # 保存模型的目录
    if not os.path.exists('./weights'): os.mkdir('./weights')

    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()

    # 一轮的步数。丢弃最后几个样本。
    train_steps = num_train // batch_size
    mixup_steps = mixup_epoch * train_steps
    cutmix_steps = cutmix_epoch * train_steps
    mosaic_steps = mosaic_epoch * train_steps
    print('\n=============== mixup and cutmix ===============')
    print('steps_per_epoch: %d' % train_steps)
    if with_mixup:
        print('mixup_steps: %d' % mixup_steps)
    else:
        print('don\'t use mixup.')
    if with_cutmix:
        print('cutmix_steps: %d' % cutmix_steps)
    else:
        print('don\'t use cutmix.')
    if with_mosaic:
        print('mosaic_steps: %d' % mosaic_steps)
    else:
        print('don\'t use mosaic.')


    nowTime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_filename = 'log%s.txt'%nowTime
    best_ap_list = [0.0, 0]  #[map, iter]
    init_iter_id = iter_id
    for _ in range(1):   # 已经被整理成1个epoch
        for step, data in enumerate(train_loader):
            if step < init_iter_id:  # 恢复训练时跳过。
                continue
            iter_id += 1

            # 估计剩余时间
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)   # time_cost=平均每一步需要多少秒
            eta_sec = (cfg.train_cfg['max_iters'] - iter_id) * time_cost   # 剩余时间=剩余步数*time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))

            # ==================== train ====================
            images = data[0]
            gt_bbox = data[1]
            gt_score = data[2]
            gt_class = data[3]
            target0 = data[4]
            target1 = data[5]
            if n_layers > 2:
                target2 = data[6]
                targets = [target0, target1, target2]
            else:
                targets = [target0, target1]
            losses = model.train_model(images, gt_bbox, gt_class, gt_score, targets)
            all_loss = 0.0
            loss_names = {}
            for loss_name in losses.keys():
                sub_loss = losses[loss_name]
                all_loss += sub_loss
                loss_names[loss_name] = sub_loss.numpy()[0]
            _all_loss = all_loss.numpy()[0]

            # 更新权重
            lr = calc_lr(iter_id, cfg)
            optimizer.set_lr(lr)
            all_loss.backward()
            if iter_id % cfg.train_cfg['update_iter'] == 0:
                optimizer.step()
                optimizer.clear_grad()
            if cfg.use_ema and iter_id % cfg.ema_iter == 0:
                ema.update()   # 更新ema字典

            # ==================== log ====================
            if iter_id % cfg.train_cfg['log_iter'] == 0:
                speed = (1.0 / time_cost)
                speed *= batch_size
                speed_msg = '%.3f imgs/s.' % (speed,)
                lr = optimizer.get_lr()
                each_loss = ''
                for loss_name in loss_names.keys():
                    loss_value = loss_names[loss_name]
                    each_loss += ' %s: %.3f,' % (loss_name, loss_value)
                strs = 'Train iter: {}, lr: {:.9f}, all_loss: {:.3f},{} eta: {}, speed: {}'.format(iter_id, lr, _all_loss, each_loss, eta, speed_msg)
                logger.info(strs)
                write(log_filename, strs)

            # ==================== save ====================
            if iter_id % cfg.train_cfg['save_iter'] == 0:
                if cfg.use_ema:
                    ema.apply()
                save_path = './weights/%d.pdparams' % iter_id
                paddle.save(model.state_dict(), save_path)
                if cfg.use_ema:
                    ema.restore()
                logger.info('Save model to {}'.format(save_path))
                write(log_filename, 'Save model to {}'.format(save_path))
                clear_model('weights')

            # ==================== eval ====================
            if iter_id % cfg.train_cfg['eval_iter'] == 0:
                if cfg.use_ema:
                    ema.apply()
                model.eval()   # 切换到验证模式
                head.set_dropblock(is_test=True)
                box_ap = eval(_decode, val_images, cfg.val_pre_path, cfg.val_path, cfg.eval_cfg['eval_batch_size'], _clsid2catid, cfg.eval_cfg['draw_image'], cfg.eval_cfg['draw_thresh'])
                logger.info("box ap: %.3f" % (box_ap[0], ))
                model.train()  # 切换到训练模式
                backbone.fix_bn()  # model.train()后需要重新固定backbone的bn层。
                head.set_dropblock(is_test=False)

                # 以box_ap作为标准
                ap = box_ap
                write(log_filename, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %.3f' % (box_ap[0], ))
                write(log_filename, 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = %.3f' % (box_ap[1], ))
                write(log_filename, 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = %.3f' % (box_ap[2], ))
                write(log_filename, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = %.3f' % (box_ap[3], ))
                write(log_filename, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = %.3f' % (box_ap[4], ))
                write(log_filename, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = %.3f' % (box_ap[5], ))
                write(log_filename, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = %.3f' % (box_ap[6], ))
                write(log_filename, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = %.3f' % (box_ap[7], ))
                write(log_filename, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %.3f' % (box_ap[8], ))
                write(log_filename, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = %.3f' % (box_ap[9], ))
                write(log_filename, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = %.3f' % (box_ap[10], ))
                write(log_filename, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = %.3f' % (box_ap[11], ))
                if ap[0] > best_ap_list[0]:
                    best_ap_list[0] = ap[0]
                    best_ap_list[1] = iter_id
                    save_path = './weights/best_model.pdparams'
                    paddle.save(model.state_dict(), save_path)
                    logger.info('Save model to {}'.format(save_path))
                    write(log_filename, 'Save model to {}'.format(save_path))
                    clear_model('weights')
                if cfg.use_ema:
                    ema.restore()
                logger.info("Best test ap: {}, in iter: {}".format(best_ap_list[0], best_ap_list[1]))
                write(log_filename, "Best test ap: {}, in iter: {}".format(best_ap_list[0], best_ap_list[1]))

        # ==================== exit ====================
        logger.info('Done.')
        write(log_filename, 'Done.')

