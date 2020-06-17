from __future__ import division

import torch
import torch.nn as nn

from .base_new import BaseDetectorNew
from .test_mixins import RPNTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import (build_assigner, bbox2roi, dbbox2roi, bbox2result, build_sampler,
                        dbbox2result, merge_aug_masks, roi2droi, mask2poly,
                        get_best_begin_point, polygonToRotRectangle_batch,
                        gt_mask_bp_obbs_list, choose_best_match_batch,
                        choose_best_Rroi_batch, dbbox_rotate_mapping, bbox_rotate_mapping)
from mmdet.core import (bbox_mapping, merge_aug_proposals, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms, merge_rotate_aug_proposals,
                        merge_rotate_aug_bboxes, multiclass_nms_rbbox)
import copy
from mmdet.core import RotBox2Polys, polygonToRotRectangle_batch
from .zzr_function import (mask2obb_mask, det_rbboxes2det_bboxes, rbboxes2rects)

@DETECTORS.register_module
class RPFPNMask(BaseDetectorNew, RPNTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 shared_head_rbbox=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 rbbox_roi_extractor=None,
                 rbbox_head=None,
                 segm_head=None,
                 atten_head=None,
                 atten_rroi_extractor=None,
                 blender=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None

        assert rbbox_roi_extractor is not None
        assert rbbox_head is not None
        super(RPFPNMask, self).__init__()

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if shared_head_rbbox is not None:
            self.shared_head_rbbox = builder.build_shared_head(shared_head_rbbox)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
        # import pdb
        # pdb.set_trace()
        if rbbox_head is not None:
            self.rbbox_roi_extractor = builder.build_roi_extractor(
                rbbox_roi_extractor)
            self.rbbox_head = builder.build_head(rbbox_head)

        self.segm_head = builder.build_head(segm_head)
        self.atten_head = builder.build_head(atten_head)
        self.atten_rroi_extractor = builder.build_roi_extractor(
                atten_rroi_extractor)
        self.blender = builder.build_head(blender)
        

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(RPFPNMask, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_shared_head_rbbox:
            self.shared_head_rbbox.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_rbbox:
            self.rbbox_roi_extractor.init_weights()
            self.rbbox_head.init_weights()

        self.segm_head.init_weights()
        self.atten_head.init_weights()
        self.atten_rroi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()

        # trans gt_masks to gt_obbs
        # gt_obbs = gt_mask_bp_obbs_list(gt_masks)
        gt_obbs = gt_mask_bp_obbs_list(mask2obb_mask(gt_bboxes, gt_masks))
        
        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)

            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals (hbb assign)
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn[0].assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn[0].sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            ## rbbox
            # rbbox_targets = self.bbox_head.get_target(
            #     sampling_results, gt_masks, gt_labels, self.train_cfg.rcnn[0])
            
            rbbox_targets = self.bbox_head.get_target(
                sampling_results, mask2obb_mask(gt_bboxes, gt_masks), gt_labels, self.train_cfg.rcnn[0])

            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *rbbox_targets)
            # losses.update(loss_bbox)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(0, name)] = (value)

        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        roi_labels = rbbox_targets[0]
        with torch.no_grad():
            # import pdb
            # pdb.set_trace()
            rotated_proposal_list = self.bbox_head.refine_rbboxes(
                roi2droi(rois), roi_labels, bbox_pred, pos_is_gts, img_meta
            )
        # import pdb
        # pdb.set_trace()
        # assign gts and sample proposals (rbb assign)
        if self.with_rbbox:
            bbox_assigner = build_assigner(self.train_cfg.rcnn[1].assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn[1].sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                gt_obbs_best_roi = choose_best_Rroi_batch(gt_obbs[i])
                assign_result = bbox_assigner.assign(
                    rotated_proposal_list[i], gt_obbs_best_roi, gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    rotated_proposal_list[i],
                    torch.from_numpy(gt_obbs_best_roi).float().to(rotated_proposal_list[i].device),
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        if self.with_rbbox:
            # (batch_ind, x_ctr, y_ctr, w, h, angle)
            rrois = dbbox2roi([res.bboxes for res in sampling_results])
            # feat enlarge
            rrois[:, 3] = rrois[:, 3] * self.rbbox_roi_extractor.w_enlarge
            rrois[:, 4] = rrois[:, 4] * self.rbbox_roi_extractor.h_enlarge
            rbbox_feats = self.rbbox_roi_extractor(x[:self.rbbox_roi_extractor.num_inputs], rrois)
            if self.with_shared_head_rbbox:
                rbbox_feats = self.shared_head_rbbox(rbbox_feats)
            cls_score, rbbox_pred = self.rbbox_head(self.rbbox_head.tower(rbbox_feats))
            rbbox_targets = self.rbbox_head.get_target_rbbox(sampling_results, gt_obbs,
                                                        gt_labels, self.train_cfg.rcnn[1])
            loss_rbbox = self.rbbox_head.loss(cls_score, rbbox_pred, *rbbox_targets)
            for name, value in loss_rbbox.items():
                losses['s{}.{}'.format(1, name)] = (value)
            
        # semantic segmentation
        segm_pred = self.segm_head(x)
        segm_label = self.segm_head.get_target(segm_pred.shape[-2:], gt_labels, gt_masks)
        loss_segm = self.segm_head.loss(segm_pred, segm_label)
        losses.update(loss_segm)
        # blender
        pos_rrois = dbbox2roi([res.pos_bboxes for res in sampling_results])
        atten_feats = self.atten_rroi_extractor(x[:self.atten_rroi_extractor.num_inputs], pos_rrois)
        atten_maps = self.atten_head(self.rbbox_head.tower(atten_feats))
        base_maps = self.blender.crop_segm(segm_pred, pos_rrois)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        mask_pred = self.blender.merge_bases(base_maps, atten_maps, pos_labels-1)
        mask_target = self.blender.get_target(sampling_results, gt_masks)
        loss_mask = self.blender.loss(mask_pred, mask_target)
        losses.update(loss_mask)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        pass
        
    
    def aug_test(self, imgs, img_metas, proposals=None, rescale=None):
        print('aug_test is not support')