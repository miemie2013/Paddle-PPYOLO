#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import os
import json
import sys
import cv2
import time
import threading
import numpy as np
import shutil
import pycocotools.mask as maskUtils
from tools.cocotools import bbox_eval, mask_eval
import logging
logger = logging.getLogger(__name__)


def multi_thread_read(j, images, _decode, offset, eval_pre_path, batch_im_id, batch_im_name, batch_img, batch_pimage, batch_ori_shape, batch_resize_shape):
    im = images[offset + j]
    im_id = im['id']
    file_name = im['file_name']
    image = cv2.imread(eval_pre_path + file_name)
    batch_im_id[j] = im_id
    batch_im_name[j] = file_name
    batch_img[j] = image
    pimage, ori_shape, resize_shape = _decode.process_image(np.copy(image))
    batch_pimage[j] = pimage
    batch_ori_shape[j] = ori_shape
    batch_resize_shape[j] = resize_shape

def read_eval_data(images,
                   _decode,
                   eval_pre_path,
                   eval_batch_size,
                   num_steps,
                   eval_dic):
    n = len(images)
    for i in range(num_steps):
        key_list = list(eval_dic.keys())
        key_len = len(key_list)
        while key_len >= 3:
            time.sleep(0.01)
            key_list = list(eval_dic.keys())
            key_len = len(key_list)


        batch_size = eval_batch_size
        if i == num_steps - 1:
            batch_size = n - (num_steps - 1) * eval_batch_size

        batch_im_id = [None] * batch_size
        batch_im_name = [None] * batch_size
        batch_img = [None] * batch_size
        batch_pimage = [None] * batch_size
        batch_ori_shape = [None] * batch_size
        batch_resize_shape = [None] * batch_size
        threads = []
        offset = i * eval_batch_size
        for j in range(batch_size):
            t = threading.Thread(target=multi_thread_read,
                                 args=(j, images, _decode, offset, eval_pre_path, batch_im_id, batch_im_name, batch_img, batch_pimage, batch_ori_shape, batch_resize_shape))
            threads.append(t)
            t.start()
        # 等待所有线程任务结束。
        for t in threads:
            t.join()

        batch_pimage = np.concatenate(batch_pimage, axis=0)
        batch_ori_shape = np.concatenate(batch_ori_shape, axis=0)
        batch_resize_shape = np.concatenate(batch_resize_shape, axis=0)
        dic = {}
        dic['batch_im_id'] = batch_im_id
        dic['batch_im_name'] = batch_im_name
        dic['batch_img'] = batch_img
        dic['batch_pimage'] = batch_pimage
        dic['batch_ori_shape'] = batch_ori_shape
        dic['batch_resize_shape'] = batch_resize_shape
        eval_dic['%.8d' % i] = dic


def write_json(j, bbox_data, mask_data, result_image, result_boxes, result_scores, result_classes, result_masks, batch_im_id, batch_im_name, _clsid2catid, draw_image, result_dir):
    image = result_image[j]
    boxes = result_boxes[j]
    scores = result_scores[j]
    classes = result_classes[j]
    masks = result_masks[j]
    if boxes is not None:
        im_id = batch_im_id[j]
        im_name = batch_im_name[j]
        n = len(boxes)
        for p in range(n):
            clsid = classes[p]
            score = scores[p]
            xmin, ymin, xmax, ymax = boxes[p]
            catid = (_clsid2catid[int(clsid)])
            w = xmax - xmin + 1
            h = ymax - ymin + 1

            bbox = [xmin, ymin, w, h]
            # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
            bbox = [round(float(x) * 10) / 10 for x in bbox]
            bbox_res = {
                'image_id': im_id,
                'category_id': catid,
                'bbox': bbox,
                'score': float(score)
            }
            bbox_data.append(bbox_res)

            segm = maskUtils.encode(np.asfortranarray(masks[p].astype(np.uint8)))
            segm['counts'] = segm['counts'].decode('utf8')

            mask_res = {
                'image_id': im_id,
                'category_id': catid,
                'segmentation': segm,
                'score': float(score)
            }
            mask_data.append(mask_res)
        if draw_image:
            cv2.imwrite('%s/images/%s' % (result_dir, im_name), image)



def eval(_decode, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image, draw_thresh, type='eval'):
    assert type in ['eval', 'test_dev']
    result_dir = 'eval_results'
    if type == 'test_dev':
        result_dir = 'results'

    # 8G内存的电脑并不能装下所有结果，所以把结果写进文件里。
    if os.path.exists('%s/bbox/' % result_dir): shutil.rmtree('%s/bbox/' % result_dir)
    if draw_image:
        if os.path.exists('%s/images/' % result_dir): shutil.rmtree('%s/images/' % result_dir)
    if not os.path.exists('%s/' % result_dir): os.mkdir('%s/' % result_dir)
    os.mkdir('%s/bbox/' % result_dir)
    if draw_image:
        os.mkdir('%s/images/' % result_dir)


    n = len(images)
    num_steps = n // eval_batch_size   # 总步数
    if n % eval_batch_size != 0:
        num_steps += 1

    logger.info('Total iter: {}'.format(num_steps))
    start = time.time()

    # 读数据的线程
    eval_dic = {}
    thr = threading.Thread(target=read_eval_data,
                           args=(images,
                                 _decode,
                                 eval_pre_path,
                                 eval_batch_size,
                                 num_steps,
                                 eval_dic))
    thr.start()
    bbox_data = []
    mask_data = []
    for i in range(num_steps):
        key_list = list(eval_dic.keys())
        key_len = len(key_list)
        while key_len == 0:
            time.sleep(0.01)
            key_list = list(eval_dic.keys())
            key_len = len(key_list)
        dic = eval_dic.pop('%.8d' % i)
        batch_im_id = dic['batch_im_id']
        batch_im_name = dic['batch_im_name']
        batch_img = dic['batch_img']
        batch_pimage = dic['batch_pimage']
        batch_ori_shape = dic['batch_ori_shape']
        batch_resize_shape = dic['batch_resize_shape']

        result_image, result_boxes, result_scores, result_classes, result_masks = _decode.detect_batch(batch_img, batch_pimage, batch_ori_shape,
                                                                                                       batch_resize_shape, draw_image=draw_image, draw_thresh=draw_thresh)
        batch_size = eval_batch_size
        if i == num_steps - 1:
            batch_size = n - (num_steps - 1) * eval_batch_size

        for j in range(batch_size):
            write_json(j, bbox_data, mask_data, result_image, result_boxes, result_scores, result_classes, result_masks, batch_im_id, batch_im_name, _clsid2catid, draw_image, result_dir)
        if (i+1) % 100 == 0:
            path = '%s/bbox/%d.json' % (result_dir, i+1)
            with open(path, 'w') as f:
                json.dump(bbox_data, f)
            bbox_data = []

            path = '%s/mask/%d.json' % (result_dir, i+1)
            with open(path, 'w') as f:
                json.dump(mask_data, f)
            mask_data = []
            logger.info('Test iter {}'.format(i+1))
    if len(bbox_data) > 0:
        path = '%s/bbox/last.json' % (result_dir, )
        with open(path, 'w') as f:
            json.dump(bbox_data, f)

        path = '%s/mask/last.json' % (result_dir, )
        with open(path, 'w') as f:
            json.dump(mask_data, f)
    logger.info('Test Done.')
    cost = time.time() - start
    logger.info('total time: {0:.6f}s'.format(cost))
    logger.info('Speed: %.6fs per image,  %.1f FPS.'%((cost / n), (n / cost)))
    if type == 'eval':
        # 开始评测
        box_ap_stats = bbox_eval(anno_file)
        mask_ap_stats = mask_eval(anno_file)
        return box_ap_stats, mask_ap_stats
    elif type == 'test_dev':
        # 生成json文件
        logger.info('Generating json file...')
        bbox_list = []
        path_dir = os.listdir('results/bbox/')
        for name in path_dir:
            with open('results/bbox/' + name, 'r', encoding='utf-8') as f2:
                for line in f2:
                    line = line.strip()
                    r_list = json.loads(line)
                    bbox_list += r_list
        # 提交到网站的文件
        with open('results/bbox_detections.json', 'w') as f2:
            json.dump(bbox_list, f2)

        mask_list = []
        path_dir = os.listdir('results/mask/')
        for name in path_dir:
            with open('results/mask/' + name, 'r', encoding='utf-8') as f2:
                for line in f2:
                    line = line.strip()
                    r_list = json.loads(line)
                    mask_list += r_list
        # 提交到网站的文件
        with open('results/mask_detections.json', 'w') as f2:
            json.dump(mask_list, f2)

        logger.info('Done.')
        return 1


