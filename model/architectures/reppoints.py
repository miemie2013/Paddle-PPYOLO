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


class RepPoints(paddle.nn.Layer):
    def __init__(self, backbone, fpn, head):
        super(RepPoints, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head

    def forward(self, x, img_metas):
        body_feats = self.backbone(x)
        body_feats, spatial_scale = self.fpn(body_feats)
        rescale = True
        outs = self.head.get_prediction(body_feats, img_metas, rescale=rescale)
        return outs

    def train_model(self, x, tag_labels, tag_bboxes, tag_centerness):
        body_feats = self.backbone(x)
        body_feats, spatial_scale = self.fpn(body_feats)
        out = self.head.get_loss(body_feats, tag_labels, tag_bboxes, tag_centerness)
        return out



