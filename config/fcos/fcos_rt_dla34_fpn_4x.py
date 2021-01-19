#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================



class FCOS_RT_DLA34_FPN_4x_Config(object):
    def __init__(self):
        # 自定义数据集
        # self.train_path = 'annotation_json/voc2012_train.json'
        # self.val_path = 'annotation_json/voc2012_val.json'
        # self.classes_path = 'data/voc_classes.txt'
        # self.train_pre_path = '../data/data4379/pascalvoc/VOCdevkit/VOC2012/JPEGImages/'   # 训练集图片相对路径
        # self.val_pre_path = '../data/data4379/pascalvoc/VOCdevkit/VOC2012/JPEGImages/'     # 验证集图片相对路径
        # self.num_classes = 20                              # 数据集类别数

        # AIStudio下的COCO数据集
        self.train_path = '../data/data7122/annotations/instances_train2017.json'
        self.val_path = '../data/data7122/annotations/instances_val2017.json'
        self.classes_path = 'data/coco_classes.txt'
        self.train_pre_path = '../data/data7122/train2017/'  # 训练集图片相对路径
        self.val_pre_path = '../data/data7122/val2017/'      # 验证集图片相对路径
        self.test_path = '../data/data7122/annotations/image_info_test-dev2017.json'      # test集
        self.test_pre_path = '../data/data7122/test2017/'    # test集图片相对路径
        self.num_classes = 80                                # 数据集类别数

        # Windows下的COCO数据集
        # self.train_path = '../COCO/annotations/instances_train2017.json'
        # self.val_path = '../COCO/annotations/instances_val2017.json'
        # self.classes_path = 'data/coco_classes.txt'
        # self.train_pre_path = '../COCO/train2017/'  # 训练集图片相对路径
        # self.val_pre_path = '../COCO/val2017/'      # 验证集图片相对路径
        # self.test_path = '../COCO/annotations/image_info_test-dev2017.json'      # test集
        # self.test_pre_path = '../COCO/test2017/'    # test集图片相对路径
        # self.num_classes = 80                       # 数据集类别数


        # ========= 一些设置 =========
        self.train_cfg = dict(
            batch_size=16,
            num_workers=5,   # 读数据的进程数
            num_threads=5,   # 读数据的线程数
            max_batch=2,     # 最大读多少个批
            model_path='dygraph_fcos_rt_dla34_fpn_4x.pdparams',
            # model_path='dygraph_r50vd_ssld.pdparams',
            # model_path='./weights/1000.pdparams',
            update_iter=1,    # 每隔几步更新一次参数
            log_iter=20,      # 每隔几步打印一次
            save_iter=1000,   # 每隔几步保存一次模型
            eval_iter=20000,   # 每隔几步计算一次eval集的mAP
            max_iters=360000,   # 训练多少步
            mixup_epoch=10,     # 前几轮进行mixup
            cutmix_epoch=10,    # 前几轮进行cutmix
            mosaic_epoch=1000,  # 前几轮进行mosaic
        )
        self.learningRate = dict(
            base_lr=0.01,
            PiecewiseDecay=dict(
                gamma=0.1,
                milestones=[300000, 340000],
            ),
            LinearWarmup=dict(
                start_factor=0.3333333333333333,
                steps=500,
            ),
        )
        self.optimizerBuilder = dict(
            optimizer=dict(
                momentum=0.9,
                type='Momentum',
            ),
            regularizer=dict(
                factor=0.0001,
                type='L2',
            ),
        )


        # 验证。用于train.py、eval.py、test_dev.py
        self.eval_cfg = dict(
            model_path='dygraph_fcos_rt_dla34_fpn_4x.pdparams',
            # model_path='./weights/1000.pdparams',
            target_size=512,
            max_size=736,
            draw_image=False,    # 是否画出验证集图片
            draw_thresh=0.15,    # 如果draw_image==True，那么只画出分数超过draw_thresh的物体的预测框。
            eval_batch_size=1,   # 验证时的批大小。
        )

        # 测试。用于demo.py
        self.test_cfg = dict(
            model_path='dygraph_fcos_rt_dla34_fpn_4x.pdparams',
            # model_path='./weights/1000.pdparams',
            target_size=512,
            max_size=736,
            draw_image=True,
            draw_thresh=0.15,   # 如果draw_image==True，那么只画出分数超过draw_thresh的物体的预测框。
        )


        # ============= 模型相关 =============
        # self.use_ema = True
        self.use_ema = False
        self.ema_decay = 0.9998
        self.ema_iter = 1
        self.backbone_type = 'DLA'
        self.backbone = dict(
            norm_type='bn',
            levels=[1, 1, 1, 2, 2, 1],
            channels=[16, 32, 64, 128, 256, 512],
            block_name='BasicBlock',
            feature_maps=[3, 4, 5],
            freeze_at=2,
            fix_bn_mean_var_at=0,
            freeze_norm=False,
            norm_decay=0.,
        )
        self.fpn_type = 'FPN'
        self.fpn = dict(
            in_channels=[512, 256, 128],
            num_chan=256,
            min_level=3,
            max_level=5,
            spatial_scale=[0.03125, 0.0625, 0.125],
            has_extra_convs=False,
            use_c5=False,
            reverse_out=False,
        )
        self.head_type = 'FCOSHead'
        self.head = dict(
            in_channel=256,
            num_classes=self.num_classes,
            fpn_stride=[8, 16, 32],
            num_convs=4,
            norm_type='gn',
            norm_reg_targets=True,
            thresh_with_ctr=True,
            centerness_on_reg=True,
            use_dcn_in_tower=False,
        )
        self.fcos_loss_type = 'FCOSLoss'
        self.fcos_loss = dict(
            loss_alpha=0.25,
            loss_gamma=2.0,
            iou_loss_type='giou',  # linear_iou/giou/iou/ciou
            reg_weights=1.0,
        )
        # self.nms_cfg = dict(
        #     nms_type='matrix_nms',
        #     score_threshold=0.01,
        #     post_threshold=0.01,
        #     nms_top_k=500,
        #     keep_top_k=100,
        #     use_gaussian=False,
        #     gaussian_sigma=2.,
        # )
        self.nms_cfg = dict(
            nms_type='multiclass_nms',
            score_threshold=0.025,
            nms_threshold=0.6,
            nms_top_k=1000,
            keep_top_k=100,
        )


        # ============= 预处理相关 =============
        self.context = {'fields': ['image', 'im_info', 'fcos_target']}
        # DecodeImage
        self.decodeImage = dict(
            to_rgb=True,
            with_mixup=False,
            with_cutmix=False,
            with_mosaic=False,
        )
        # RandomFlipImage
        self.randomFlipImage = dict(
            prob=0.5,
        )
        # NormalizeImage
        self.normalizeImage = dict(
            is_channel_first=False,
            is_scale=False,
            mean=[123.675, 116.28, 103.53],
            std=[1.0, 1.0, 1.0],
        )
        # ResizeImage
        # 图片短的那一边缩放到选中的target_size，长的那一边等比例缩放；如果这时候长的那一边大于max_size，
        # 那么改成长的那一边缩放到max_size，短的那一边等比例缩放。这时候im_scale_x = im_scale， im_scale_y = im_scale。
        # resize_box=True 表示真实框（格式是x0y0x1y1）也跟着缩放，横纵坐标分别乘以im_scale_x、im_scale_y。
        # resize_box=False表示真实框（格式是x0y0x1y1）不跟着缩放，因为后面会在Gt2FCOSTarget中缩放。
        self.resizeImage = dict(
            target_size=[256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
            max_size=900,
            interp=1,
            use_cv2=True,
            resize_box=False,
        )
        # Permute
        self.permute = dict(
            to_bgr=False,
            channel_first=True,
        )
        # PadBatch
        self.padBatch = dict(
            pad_to_stride=32,   # 添加黑边使得图片边长能够被pad_to_stride整除。pad_to_stride代表着最大下采样倍率，这个模型最大到p5，为32。
            use_padded_im_info=False,
        )
        # Gt2FCOSTarget
        self.gt2FCOSTarget = dict(
            object_sizes_boundary=[64, 128],
            center_sampling_radius=1.5,
            downsample_ratios=[8, 16, 32],
            norm_reg_targets=True,
        )

        # 预处理顺序。增加一些数据增强时这里也要加上，否则train.py中相当于没加！
        self.sample_transforms_seq = []
        self.sample_transforms_seq.append('decodeImage')
        self.sample_transforms_seq.append('randomFlipImage')
        self.sample_transforms_seq.append('normalizeImage')
        self.sample_transforms_seq.append('resizeImage')
        self.sample_transforms_seq.append('permute')
        self.batch_transforms_seq = []
        self.batch_transforms_seq.append('padBatch')
        self.batch_transforms_seq.append('gt2FCOSTarget')


