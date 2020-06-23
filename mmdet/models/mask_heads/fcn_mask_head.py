import cv2
import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from mmdet.ops.roi_align_rotated import roi_align_rotated_cuda
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import (mask_target, mask_target_hbb, 
                        mask_rotate_target, mask_rotate_adaptive_target, 
                        mask_rotate_fix_target)


@HEADS.register_module
class FCNMaskHead(nn.Module):

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
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(FCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_mask = build_loss(loss_mask)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
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
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def get_target_hbb(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets_hbb = mask_target_hbb(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets_hbb

    def get_rotate_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_rotate_targets = mask_rotate_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_rotate_targets

    # def get_rotate_target(self, sampling_results, gt_masks_list, rcnn_train_cfg):
    #     pos_proposals_list = [res.pos_bboxes for res in sampling_results]
    #     pos_assigned_gt_inds_list = [
    #         res.pos_assigned_gt_inds for res in sampling_results
    #     ]
    #     mask_size = rcnn_train_cfg.mask_size
    #     mask_targets_list = []
    #     for pos_proposals, pos_assigned_gt_inds, gt_masks \
    #         in zip(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list):
    #         num_pos = pos_proposals.size(0)
    #         mask_targets = []
    #         if num_pos > 0:
    #             for i in range(num_pos):
    #                 gt_mask = gt_masks[pos_assigned_gt_inds[i]]
    #                 gt_mask = torch.from_numpy(gt_mask[np.newaxis, \
    #                     np.newaxis, :, :]).float().to(pos_proposals.device)
    #                 x,y,w,h,theta = pos_proposals[i, :]
    #                 roi = torch.tensor([[0.0, x, y, w, h, theta]]).to(pos_proposals.device)
    #                 target = pos_proposals.new_zeros(1, 1, mask_size, mask_size)
    #                 roi_align_rotated_cuda.forward(gt_mask, roi, mask_size, mask_size, 1.0, 2, target)
    #                 mask_targets.append(target.squeeze())
    #                 # mask_targets.append(torch.where(target>0.5,target.new_full(target.shape, 1),target.new_full(target.shape, 0)))
    #         else:
    #             mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
    #         mask_targets_list.append(torch.stack(mask_targets))
    #     return torch.cat(mask_targets_list)

    def get_rotate_fix_target(self, sampling_results, gt_masks, 
                                rcnn_train_cfg, out_w, out_h):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_rotate_fix_targets = mask_rotate_fix_target(pos_proposals, 
                                   pos_assigned_gt_inds, gt_masks, 
                                   rcnn_train_cfg, out_w*2, out_h*2)
        return mask_rotate_fix_targets

    def get_rotate_adaptive_target(self, sampling_results, gt_masks, rcnn_train_cfg, ratio_max):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_rotate_adaptive_targets = mask_rotate_adaptive_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg, ratio_max)
        return mask_rotate_adaptive_targets

    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            if not isinstance(scale_factor, (float, np.ndarray)):
                scale_factor = scale_factor.cpu().numpy()
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            bbox[0] = bbox[0].clip(0, img_w-1)
            bbox[1] = bbox[1].clip(0, img_h-1)
            bbox[2] = bbox[2].clip(0, img_w-1)
            bbox[3] = bbox[3].clip(0, img_h-1)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)

            if rcnn_test_cfg.get('crop_mask', False):
                im_mask = bbox_mask
            else:
                im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

            if rcnn_test_cfg.get('rle_mask_encode', True):
                rle = mask_util.encode(
                    np.array(im_mask[:, :, np.newaxis], order='F'))[0]
                cls_segms[label - 1].append(rle)
            else:
                cls_segms[label - 1].append(im_mask)

        return cls_segms

    def get_rotate_seg_masks(self, mask_pred, det_rects, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        rects  = det_rects.cpu().numpy()[:, :5]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(rects.shape[0]):
            rects[i, :4] = rects[i, :4] / scale_factor
            x,y,w,h,theta = rects[i,:]
            theta = -1*theta*180/np.pi
            w = int(np.maximum(w, 1))
            h = int(np.maximum(h, 1))
            label = labels[i]
            
            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
                
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            expand_layer = np.zeros((img_h*2,img_w*2),dtype='uint8')
            w_start = int(img_w / 2)
            h_start = int(img_h / 2)
            
            rbbox_mask = mmcv.imresize(mask_pred_, (w, h))
            rbbox_mask = (rbbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            
            expand_layer[:h, :w] = rbbox_mask
            M = np.float32([[1,0,x - w/2 + w_start], [0, 1, y - h/2 + h_start]])
            expand_layer = cv2.warpAffine(expand_layer,M,expand_layer.shape[::-1], borderValue=0)
            M = cv2.getRotationMatrix2D((x+w_start,y+h_start),theta,1)
            expand_layer = cv2.warpAffine(expand_layer,M,expand_layer.shape[::-1],borderValue=0)
            
            im_mask = expand_layer[h_start:h_start+img_h, w_start:w_start+img_w]
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms
