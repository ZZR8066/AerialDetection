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

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from torch import sqrt

@HEADS.register_module
class PFPNHead(nn.Module):

    def __init__(self,
                num_levels=4,
                num_bases=4,
                basis_stride=4,
                num_classes=16,
                planes=128,
                in_channels=256,
                segm_loss_weight=0.3,
                common_stride=4,
                feature_strides=[4,8,16,32]):

        super(PFPNHead, self).__init__()
        self.num_bases = num_bases
        self.basis_stride = basis_stride
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.segm_loss_weight = segm_loss_weight
        self.feature_strides = feature_strides
        self.common_stride = common_stride
        
        self.refine = nn.ModuleList()
        for i in range(num_levels):
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[i]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                conv = ConvModule(
                in_channels if k ==0 else planes,
                out_channels=planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=None,
                norm_cfg=dict(type='SyncBN',requires_grad=True),
                activation='relu',
                inplace=True,
                activate_last=True)
                head_ops.append(conv)
                if feature_strides[i] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.refine.append(nn.Sequential(*head_ops))
            
        self.tower = nn.ModuleList()
        for i in range(num_levels):
            l_conv = ConvModule(
                in_channels=planes,
                out_channels=planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=None,
                norm_cfg=dict(type='SyncBN',requires_grad=True),
                activation='relu',
                inplace=True,
                activate_last=True)
            self.tower.append(l_conv)

        self.tower.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.tower.append(ConvModule(
                            in_channels=planes,
                            out_channels=planes,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                            conv_cfg=None,
                            norm_cfg=dict(type='SyncBN',requires_grad=True),
                            activation='relu',
                            inplace=True,
                            activate_last=True))
        self.tower.append(nn.Conv2d(planes, num_bases, 1))
        self.tower = nn.Sequential(*self.tower)
        
        self.seg_head = nn.Sequential(nn.Conv2d(in_channels, planes, kernel_size=3,
                                                stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(planes),
                                      nn.ReLU(),
                                      nn.Conv2d(planes, planes, kernel_size=3,
                                                stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(planes),
                                      nn.ReLU(),
                                      nn.Conv2d(planes, num_classes, kernel_size=1,
                                                stride=1))
        
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
    
    def forward(self, features):
        for i in range(self.num_levels):
            if i == 0:
                x = self.refine[i](features[i])
            else:
                xp = self.refine[i](features[i])
                x = x + xp
        bases_out = self.tower(x)
        return bases_out

    def add_segm_loss(self, features, img_shape, gt_labels_list, gt_masks_list):
        loss = dict()
        sem_out = self.seg_head(features[0])
        gt_sem = self.get_segm_target(img_shape, gt_labels_list, gt_masks_list)
        gt_sem = gt_sem.unsqueeze(1).float()
        gt_sem = F.interpolate(
                gt_sem, scale_factor=1 / self.basis_stride)
        seg_loss = F.cross_entropy(
                sem_out, gt_sem.squeeze(1).long())
        loss['loss_basis_sem'] = seg_loss * self.segm_loss_weight
        return loss
    
    def get_segm_target(self, img_shape, gt_labels_list, gt_masks_list):
        segm_targets_list = []
        for gt_labels, gt_masks in zip(gt_labels_list, gt_masks_list):
            segm_target = np.zeros(img_shape,dtype='uint8')
            for label in range(0,self.num_classes):
                flag = (gt_labels == label).cpu().numpy()
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