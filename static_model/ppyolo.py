#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================


class PPYOLO(object):
    def __init__(self, backbone, head):
        super(PPYOLO, self).__init__()
        self.backbone = backbone
        self.head = head

    def __call__(self, x, im_size):
        body_feats = self.backbone(x)
        out = self.head.get_prediction(body_feats, im_size)
        return out



