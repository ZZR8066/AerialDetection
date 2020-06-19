import cv2
import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule

from .fcn_mask_head import FCNMaskHead

@HEADS.register_module
class AttenFCNLossHead(FCNMaskHead):
    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                loss_atten_weight=0.3):
        super(AttenFCNLossHead, self).__init__(
                num_convs,
                roi_feat_size,
                in_channels,
                conv_kernel_size,
                conv_out_channels,
                upsample_method,
                upsample_ratio,
                num_classes,
                class_agnostic,
                conv_cfg,
                norm_cfg,
                loss_mask)
        self.loss_atten_weight = loss_atten_weight
        self.conv_mask = nn.Conv2d(self.conv_out_channels, 1, 1)

    def init_weights(self):
        for m in [self.upsample, self.conv_logits, self.conv_mask]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        atten_maps = self.conv_logits(x)
        atten_pred = self.conv_mask(x)
        return atten_pred, atten_maps

    def loss(self, pred, target):
        loss = dict()
        loss['loss_atten'] = self.loss_atten_weight * F.binary_cross_entropy_with_logits(pred, \
                target, reduction='mean')[None]
        return loss