import cv2
import mmcv
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init, xavier_init
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class BlendMaskHead(nn.Module):

    def __init__(self,
                 # Bottom Module Config
                 bottom_in_channel=256,
                 bottom_out_channel=256,
                 up_in_channel=256,
                 up_out_channel=256,
                 up_sample_method='bilinear',
                 up_sample_ratio=4,
                 num_bottom_module_convs = 1,
                 bottom_module_kernel_size = 3,
                 K=4,
                 M=14,
                 R=56,
                 # Attentation Module Config
                 num_attentation_module_convs = 5,
                 # other config
                 num_classes=81,
                 class_agnostic=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(BlendMaskHead, self).__init__()
        self.bottom_in_channel = bottom_in_channel
        self.bottom_out_channel = bottom_out_channel
        self.up_in_channel = up_in_channel
        self.up_out_channel = up_out_channel
        self.up_sample_method = up_sample_method
        self.up_sample_ratio = up_sample_ratio
        self.num_bottom_module_convs = num_bottom_module_convs
        self.bottom_module_kernel_size = bottom_module_kernel_size
        self.K = K
        self.M = M
        self.R = R
        self.num_attentation_module_convs = num_attentation_module_convs
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.loss_mask = build_loss(loss_mask)
        
        # bottom module 
        self.bottom_conv = ConvModule(
                bottom_in_channel,
                bottom_out_channel,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
        
        self.up_conv = ConvModule(
                up_in_channel,
                up_out_channel,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

        self.bottom_module_convs = nn.ModuleList()
        padding = (self.bottom_module_kernel_size - 1) // 2
        for i in range(self.num_bottom_module_convs - 1):
            self.bottom_module_convs.append(
                ConvModule(
                    bottom_out_channel + up_out_channel,
                    bottom_out_channel + up_out_channel,
                    self.bottom_module_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        self.bottom_module_convs.append(
            ConvModule(
                bottom_out_channel + up_out_channel,
                K,
                self.bottom_module_kernel_size,
                padding=padding,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))
        # Attentation Module
        self.attentation_module_convs = nn.ModuleList()
        for i in range(num_attentation_module_convs):
            self.attentation_module_convs.append(
                ConvModule(
                    bottom_in_channel,
                    K*M*M,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        bottom_feat = x[0]
        up_feat = x[2]
        # Bottom Module
        bottom_feat = self.bottom_conv(bottom_feat)
        up_feat = self.up_conv(up_feat)
        up_feat = F.interpolate(up_feat,
            scale_factor=self.up_sample_ratio, mode=self.up_sample_method)
        bottom_module_output = torch.cat((bottom_feat, up_feat), 1)
        for conv in self.bottom_module_convs:
            bottom_module_output = conv(bottom_module_output)
        # Attentation Module
        attentation_module_output = []
        for i, conv in enumerate(self.attentation_module_convs):
            attentation_module_output.append(conv(x[i]))
        return bottom_module_output, attentation_module_output
            
    def get_correspond_atten(self, len_ranges, strides, attentation_feat, bboxes_list):
        final_feat = []
        atten_size = attentation_feat[0].shape[1]
        for i, bboxes in enumerate(bboxes_list):
            atten = bboxes.new_zeros((bboxes.shape[0], atten_size))
            xc = (bboxes[:, 2] + bboxes[:, 0]) / 2
            yc = (bboxes[:, 3] + bboxes[:, 1]) / 2
            w = bboxes[:, 2] - bboxes[:, 0] + 1
            h = bboxes[:, 3] - bboxes[:, 1] + 1
            wh = torch.stack((w, h),-1)
            wh = torch.max(wh, -1)[0]
            assert wh.min() >= 0
            assert wh.max() <= 1e8
            levels = wh.new_full(wh.shape, -1)
            for level, len_range in enumerate(len_ranges):
                low_limit = len_range[0]
                high_limit = len_range[1]
                valid = torch.mul(wh>=low_limit, wh<high_limit)
                levels[valid] = level
            for level in range(len(len_ranges)):
                inds = levels == level
                if inds.any():
                    x = (xc[inds] / strides[level]).long()
                    y = (yc[inds] / strides[level]).long()
                    atten[inds] = attentation_feat[level][i, :, y, x].permute(1,0)
            final_feat.append(atten)
        return final_feat

    def blender(self, bottom_feat, atten_feat):
        atten_feat = torch.cat(atten_feat)
        atten_feat = atten_feat.view(-1, self.K, self.M, self.M)
        atten_feat = F.interpolate(atten_feat,
                scale_factor=(self.R / self.M), mode=self.up_sample_method)
        atten_feat = atten_feat.view(-1, self.K, self.R * self.R)
        atten_feat = F.softmax(atten_feat, dim=-1)
        bottom_feat = bottom_feat.view(-1, self.K, self.R * self.R)
        blend_mask = bottom_feat * atten_feat
        blend_mask = blend_mask.sum(1)
        blend_mask = blend_mask.view(-1, self.R, self.R)
        return blend_mask

    def get_mask_target(self, gt_bboxes_list, gt_masks_list):
        mask_targets_list = []
        mask_size = self.R
        for gt_bboxes, gt_masks in zip(gt_bboxes_list, gt_masks_list):
            num_pos = gt_bboxes.size(0)
            mask_targets = []
            if num_pos > 0:
                gt_bboxes_np = gt_bboxes.cpu().numpy()
                for i in range(num_pos):
                    gt_mask = gt_masks[i]
                    bbox = gt_bboxes_np[i, :].astype(np.int32)
                    x1, y1, x2, y2 = bbox
                    w = np.maximum(x2 - x1 + 1, 1)
                    h = np.maximum(y2 - y1 + 1, 1)
                    # mask is uint8 both before and after resizing
                    target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                        (mask_size, mask_size))
                    mask_targets.append(target)
                mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
                    gt_bboxes.device)
            else:
                mask_targets = gt_bboxes.new_zeros((0, mask_size, mask_size))
            mask_targets_list.append(mask_targets)
        return torch.cat(mask_targets_list)

    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss