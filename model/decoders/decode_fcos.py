#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import colorsys
import threading
import paddle

from tools.transform import *


class Decode_FCOS(object):
    def __init__(self, model, all_classes, place, cfg, for_test=True):
        self.all_classes = all_classes
        self.num_classes = len(self.all_classes)
        self.model = model
        self.place = place

        # 图片预处理
        self.context = cfg.context
        # sample_transforms
        self.to_rgb = cfg.decodeImage['to_rgb']
        self.normalizeImage = NormalizeImage(**cfg.normalizeImage)
        target_size = cfg.eval_cfg['target_size']
        max_size = cfg.eval_cfg['max_size']
        if for_test:
            target_size = cfg.test_cfg['target_size']
            max_size = cfg.test_cfg['max_size']
        self.resizeImage = ResizeImage(target_size=target_size, resize_box=False, interp=cfg.resizeImage['interp'],
                                       max_size=max_size, use_cv2=cfg.resizeImage['use_cv2'])
        self.permute = Permute(**cfg.permute)
        # batch_transforms
        self.padBatch = PadBatch(use_padded_im_info=True, pad_to_stride=cfg.padBatchSingle['pad_to_stride'])
        self.padBatchSingle = PadBatchSingle(use_padded_im_info=True)


    # 处理一张图片
    def detect_image(self, image, pimage, im_info, draw_image, draw_thresh=0.0):
        pred = self.predict(pimage, im_info)
        no_obj = False
        if len(pred[0]) == 0:
            no_obj = True
        elif np.sum(pred[0]) < 0.0:
            no_obj = True
        if no_obj:
            boxes = np.array([])
            classes = np.array([])
            scores = np.array([])
        else:
            boxes = pred[0][:, 2:]
            scores = pred[0][:, 1]
            classes = pred[0][:, 0].astype(np.int32)
        if len(scores) > 0 and draw_image:
            pos = np.where(scores >= draw_thresh)
            boxes2 = boxes[pos]         # [M, 4]
            scores2 = scores[pos]       # [M, ]
            classes2 = classes[pos]     # [M, ]
            self.draw(image, boxes2, scores2, classes2)
        return image, boxes, scores, classes

    # 处理一张图片
    def get_heatmap(self, image, pimage, im_info, draw_image, draw_thresh=0.0):
        pimage = paddle.to_tensor(pimage, place=self.place)
        im_info = paddle.to_tensor(im_info, place=self.place)
        pred_scores, pred_ltrb, pred_loc = self.model(pimage, im_info, get_heatmap=True)
        pred_scores = [pred.numpy() for pred in pred_scores]
        pred_ltrb = [pred.numpy() for pred in pred_ltrb]
        pred_loc = [pred.numpy() for pred in pred_loc]

        n_feat = len(pred_scores)
        fpn_images = []
        thr = 0.05
        mask_alpha = 0.45
        line_color = (0, 0, 0)
        color = (0, 0, 255)
        color = np.array(color).astype(np.float32)
        color = np.reshape(color, (1, 1, 3))
        line_thick = 1

        # 图片放大好观察
        scales = [1.0, 2.0, 5.0, 5.0, 10.0]
        if n_feat == 3:
            scales = [3.0, 7.0, 13.0]

        image_h, image_w, _ = image.shape
        for i in range(n_feat):
            fpn_image = np.copy(image)

            scale = scales[i]


            fpn_image = cv2.resize(
                fpn_image,
                None,
                None,
                fx=scale,
                fy=scale,
                interpolation=1)


            scores = pred_scores[i]
            ltrb = pred_ltrb[i]
            loc = pred_loc[i]
            loc *= scale
            ltrb *= scale

            stride = loc[0, 0, 0, 0] * 2.0
            # N, H, W, 4 = boxes.shape
            N, C, H, W = scores.shape
            for j in range(H):
                for k in range(W):
                    # 先画格子分界线。
                    grid_rb_x = loc[0, j, k, 0] + stride * 0.5
                    grid_rb_x = int(grid_rb_x)
                    grid_rb_y = loc[0, j, k, 1] + stride * 0.5
                    grid_rb_y = int(grid_rb_y)
                    cv2.line(fpn_image, (grid_rb_x, int(grid_rb_y - stride)), (grid_rb_x, grid_rb_y), line_color, line_thick)
                    cv2.line(fpn_image, (int(grid_rb_x - stride), grid_rb_y), (grid_rb_x, grid_rb_y), line_color, line_thick)

                    _score = scores[0, :, j, k]   # [80, ]
                    _ltrb  =   ltrb[0, j, k, :]   # [4, ]
                    _loc   =    loc[0, j, k, :]   # [2, ]
                    max_score = np.max(_score)
                    if max_score < thr:
                        continue
                    else:
                        pos = np.where(_score >= thr)[0]
                        for p in pos:
                            # 类别id + 分数 写上去
                            bbox_mess = '%s: %.2f' % (self.all_classes[p], _score[p])
                            left = _loc[0] - stride * 0.5
                            top = _loc[1] - stride * 0.5
                            right = _loc[0] + stride * 0.5
                            bottom = _loc[1] + stride * 0.5

                            left = int(left)
                            top = int(top)
                            right = int(right)
                            bottom = int(bottom)

                            cv2.putText(fpn_image, bbox_mess, (left, top+9), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.3, (0, 0, 0), 1, lineType=cv2.LINE_AA)

                            # 画分数的热力图
                            color2 = np.copy(color)
                            color2 *= _score[p]
                            target_region = fpn_image[top:bottom, left:right, :]
                            target_ms = np.ones(target_region.shape, np.float32)
                            target_region = target_ms * (target_region * (1 - mask_alpha) + color2 * mask_alpha) + (
                                    1 - target_ms) * target_region
                            fpn_image[top:bottom, left:right, :] = target_region
            fpn_images.append(fpn_image)
        return fpn_images

    # 多线程后处理
    def multi_thread_post(self, i, pred, result_image, result_boxes, result_scores, result_classes, batch_img, draw_image, draw_thresh):
        no_obj = False
        if len(pred[i]) == 0:
            no_obj = True
        elif np.sum(pred[i]) < 0.0:
            no_obj = True
        if no_obj:
            boxes = np.array([])
            classes = np.array([])
            scores = np.array([])
        else:
            boxes = pred[i][:, 2:]
            scores = pred[i][:, 1]
            classes = pred[i][:, 0].astype(np.int32)
        if len(scores) > 0 and draw_image:
            pos = np.where(scores >= draw_thresh)
            boxes2 = boxes[pos]      # [M, 4]
            scores2 = scores[pos]    # [M, ]
            classes2 = classes[pos]  # [M, ]
            self.draw(batch_img[i], boxes2, scores2, classes2)
        result_image[i] = batch_img[i]
        result_boxes[i] = boxes
        result_scores[i] = scores
        result_classes[i] = classes

    # 处理一批图片
    def detect_batch(self, batch_img, batch_pimage, batch_im_info, draw_image, draw_thresh=0.0):
        batch_size = len(batch_img)
        result_image, result_boxes, result_scores, result_classes = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size

        pred = self.predict(batch_pimage, batch_im_info)   # [bs, M, 6]

        threads = []
        for i in range(batch_size):
            t = threading.Thread(target=self.multi_thread_post,
                                 args=(i, pred, result_image, result_boxes, result_scores, result_classes, batch_img, draw_image, draw_thresh))
            threads.append(t)
            t.start()
        # 等待所有线程任务结束。
        for t in threads:
            t.join()
        return result_image, result_boxes, result_scores, result_classes

    def draw(self, image, boxes, scores, classes):
        image_h, image_w, _ = image.shape
        # 定义颜色
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for box, score, cl in zip(boxes, scores, classes):
            x0, y0, x1, y1 = box
            left = max(0, np.floor(x0 + 0.5).astype(int))
            top = max(0, np.floor(y0 + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
            bbox_color = colors[cl]
            # bbox_thick = 1 if min(image_h, image_w) < 400 else 2
            bbox_thick = 1
            cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
            bbox_mess = '%s: %.2f' % (self.all_classes[cl], score)
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
            cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    def process_image(self, img, batch_transforms=True):
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        context = self.context
        sample = {}
        sample['image'] = img
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]

        sample = self.normalizeImage(sample, context)
        sample = self.resizeImage(sample, context)
        sample = self.permute(sample, context)

        if batch_transforms:
            # batch_transforms
            samples = self.padBatch([sample], context)
            sample = samples[0]

            pimage = np.expand_dims(sample['image'], axis=0)
            im_info = np.expand_dims(sample['im_info'], axis=0)
            return pimage, im_info
        else:
            return sample

    def process_image_batch_transforms(self, sample, max_shape):
        sample = self.padBatchSingle(max_shape, sample, self.context)

        pimage = np.expand_dims(sample['image'], axis=0)
        im_info = np.expand_dims(sample['im_info'], axis=0)
        return pimage, im_info

    def predict(self, image, im_info):
        image = paddle.to_tensor(image, place=self.place)
        im_info = paddle.to_tensor(im_info, place=self.place)
        preds = self.model(image, im_info)
        preds = [pred.numpy() for pred in preds]
        return preds



