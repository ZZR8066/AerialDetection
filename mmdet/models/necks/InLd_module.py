import cv2
import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.ops import DeformConv, ModulatedDeformConv, ContextBlock
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer

from mmdet.core import multi_apply

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from torch import sqrt
from ..registry import NECKS

@NECKS.register_module
class InLD_Module(nn.Module):

    def __init__(self,
                strides=[8, 16, 32, 64, 128], # The stride of lowest layer
                stacked_convs=7,
                dilations=[1,1,2,4,8,16,1],
                num_classes=17, # Class num + bg
                in_channels=256,
                segm_loss_weight=[0.2, 0.2, 0.2, 0.2, 0.2]):
        super(InLD_Module, self).__init__()
        self.strides = strides
        self.num_classes = num_classes
        self.segm_loss_weight = segm_loss_weight
        self.stacked_convs =  stacked_convs
        
        self.dilated_convs_levels = nn.ModuleList()
        for _ in range(len(strides)):
            dilated_convs_level = nn.ModuleList()
            dilated_convs = nn.ModuleList()
            for dilation in dilations:
                d_conv = ConvModule(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    dilation=dilation,
                    padding=dilation,
                    stride=1,
                    bias=False,
                    conv_cfg=None,
                    norm_cfg=dict(type='SyncBN',requires_grad=True),
                    activation='relu',
                    inplace=True,
                    activate_last=True)
                dilated_convs.append(d_conv)

            dilated_convs = nn.Sequential(*dilated_convs)
            dilated_convs_level.append(dilated_convs)
            InLd_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
            dilated_convs_level.append(InLd_conv)
            segm_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1)
            dilated_convs_level.append(segm_conv)

            self.dilated_convs_levels.append(dilated_convs_level)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(torch.tensor(2.) / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.dilated_convs_levels, train=True)

    def forward_single(self, feat, dilated_convs_level, train):
        orign_feat = feat

        # denoise
        feat = dilated_convs_level[0](feat)

        denoise_feat = dilated_convs_level[1](feat)
        # denoise_feat = orign_feat * denoise_feat.softmax(dim=1)
        denoise_feat = orign_feat * (denoise_feat.sigmoid())

        if train:
            segm_pred = dilated_convs_level[2](feat)
            return denoise_feat, segm_pred
        else:
            return denoise_feat
        '''
        # denoise
        feat = self.dilated_convs(feat)
        
        denoise_feat = self.InLd_conv(feat)
        denoise_feat = orign_feat * denoise_feat
        
        if train:
            segm_pred = self.segm_conv(feat)
            return denoise_feat, segm_pred
        else:
            return denoise_feat
        '''

    def add_segm_loss(self, segm_preds, gt_labels_list, gt_masks_list):
        img_shape = tuple([ l * self.strides[0] for l in segm_preds[0].shape[-2:]])
        gt_segm = self.get_segm_target(img_shape, gt_labels_list, gt_masks_list)
        gt_segm = gt_segm.unsqueeze(1).float()
        gt_segm = [gt_segm for _ in range(len(segm_preds))]
        segm_loss, _ = multi_apply(self.add_segm_loss_single, segm_preds, gt_segm, self.strides, self.segm_loss_weight)
        return dict(loss_segm=segm_loss)
    
    def add_segm_loss_single(self, segm_pred, gt_segm, stride, segm_loss_weight):
        gt_segm = F.interpolate(
                gt_segm, scale_factor=1 / stride)
        segm_loss = F.cross_entropy(
                segm_pred, gt_segm.squeeze(1).long())
        segm_loss = segm_loss_weight * segm_loss
        return segm_loss, None
        
    def get_segm_target(self, img_shape, gt_labels_list, gt_masks_list):
        segm_targets_list = []
        for gt_labels, gt_masks in zip(gt_labels_list, gt_masks_list):
            segm_target = np.zeros(img_shape,dtype='uint8')
            for label in range(0,self.num_classes):
                flag = (gt_labels == label).cpu().numpy().astype(np.bool)
                if flag.any():
                    assert label != 0
                    masks = gt_masks[flag]
                    for mask in masks:
                        mask = np.pad(mask,((0, img_shape[0]-mask.shape[0]), \
                                            (0, img_shape[1]-mask.shape[1])),
                                            'constant', constant_values = (0,0))
                        mask = mask.astype(np.bool)
                        segm_target[mask] = label
            segm_targets_list.append(torch.from_numpy(segm_target).unsqueeze(0).long().to(gt_labels.device))
        return torch.cat(list(segm_targets_list))