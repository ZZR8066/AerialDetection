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
class RotateBlendMaskRCNN(BaseDetectorNew, RPNTestMixin):

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
                 basis_head=None,
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
        super(RotateBlendMaskRCNN, self).__init__()

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
        
        self.basis_head = builder.build_head(basis_head)
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
        super(RotateBlendMaskRCNN, self).init_weights(pretrained)
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
        
        self.basis_head.init_weights()
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
        gt_obb_masks = mask2obb_mask(gt_bboxes, gt_masks, enlarge_pixel=2)
        gt_obbs = gt_mask_bp_obbs_list(gt_obb_masks)
        
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
                sampling_results,gt_obb_masks, gt_labels, self.train_cfg.rcnn[0])

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
            # rrois[:, 3] = rrois[:, 3] * 1.2
            # rrois[:, 4] = rrois[:, 4] * 1.4
            rrois[:, 3] = rrois[:, 3] * self.rbbox_roi_extractor.w_enlarge
            rrois[:, 4] = rrois[:, 4] * self.rbbox_roi_extractor.h_enlarge
            rbbox_feats = self.rbbox_roi_extractor(x[:self.rbbox_roi_extractor.num_inputs],
                                                   rrois)
            if self.with_shared_head_rbbox:
                rbbox_feats = self.shared_head_rbbox(rbbox_feats)
            cls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
            rbbox_targets = self.rbbox_head.get_target_rbbox(sampling_results, gt_obbs,
                                                        gt_labels, self.train_cfg.rcnn[1])
            loss_rbbox = self.rbbox_head.loss(cls_score, rbbox_pred, *rbbox_targets)
            for name, value in loss_rbbox.items():
                losses['s{}.{}'.format(1, name)] = (value)
                
        # basis pred
        basis_pred = self.basis_head.forward(x)
        loss_segm = self.basis_head.add_segm_loss(x, (int(basis_pred.shape[-2] * self.basis_head.basis_stride / 2), \
            int(basis_pred.shape[-1] * self.basis_head.basis_stride / 2)), gt_labels, gt_masks)
        losses.update(loss_segm)

        # atten pred
        # pos_rrois = dbbox2roi(
        #     [res.pos_bboxes for res in sampling_results])
        # atten_feats = self.atten_rroi_extractor(
        #             x[:self.atten_rroi_extractor.num_inputs], pos_rrois)
        # atten_maps = self.atten_head(atten_feats)
        # base_maps = self.blender.crop_segm(basis_pred, pos_rrois)
        # mask_pred = self.blender.merge_bases(base_maps, atten_maps)
        # mask_targets = self.atten_head.get_rotate_target(
        #         sampling_results, gt_masks, self.train_cfg.rcnn[1])
        # loss_mask = self.blender.loss(mask_pred, mask_targets)
        # losses.update(loss_mask)

        # atten pred
        pos_rrois = dbbox2roi(
            [res.pos_bboxes for res in sampling_results])
        atten_feats = self.atten_rroi_extractor(
                    x[:self.atten_rroi_extractor.num_inputs], pos_rrois)
        atten_pred, atten_maps = self.atten_head(atten_feats)
        # add atten loss
        atten_target = self.atten_head.get_rotate_target(
                sampling_results, gt_masks, self.train_cfg.rcnn[0])
        loss_atten = self.atten_head.loss(atten_pred.squeeze(1), atten_target)
        losses.update(loss_atten)
        # mask loss
        base_maps = self.blender.crop_segm(basis_pred, pos_rrois)
        mask_pred = self.blender.merge_bases(base_maps, atten_maps)
        mask_targets = self.atten_head.get_rotate_target(
                sampling_results, gt_masks, self.train_cfg.rcnn[1])
        loss_mask = self.blender.loss(mask_pred, mask_targets)
        losses.update(loss_mask)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']

        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_label = cls_score.argmax(dim=1)
        rrois = self.bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label, bbox_pred,
                                                      img_meta[0])

        rrois_enlarge = copy.deepcopy(rrois)
        rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor.w_enlarge
        rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor.h_enlarge
        rbbox_feats = self.rbbox_roi_extractor(
            x[:len(self.rbbox_roi_extractor.featmap_strides)], rrois_enlarge)
        if self.with_shared_head_rbbox:
            rbbox_feats = self.shared_head_rbbox(rbbox_feats)

        rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
        det_rbboxes, det_labels = self.rbbox_head.get_det_rbboxes(
            rrois,
            rcls_score,
            rbbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        rbbox_results = dbbox2result(det_rbboxes, det_labels,
                                     self.rbbox_head.num_classes)
        
        det_bboxes = det_rbboxes2det_bboxes(det_rbboxes)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                     self.rbbox_head.num_classes)
        
        segm_results = self.simple_test_rotate_mask(x, img_meta, 
                        det_rbboxes, det_labels, rescale=rescale)
        return bbox_results, rbbox_results, segm_results
        
    # def simple_test_rotate_mask(self, x, img_meta, 
    #         det_rbboxes, det_labels, rescale=False):
    #     ori_shape = img_meta[0]['ori_shape']
    #     scale_factor = img_meta[0]['scale_factor']
    #     if det_rbboxes.shape[0] == 0:
    #         segm_result = [[] for _ in range(self.blender.num_classes - 1)]
    #     else:
    #         # basis pred 
    #         basis_pred = self.basis_head.forward(x)
    #         # atten pred
    #         # if det_bboxes is rescaled to the original image size, we need to
    #         # rescale it back to the testing scale to obtain RoIs.
    #         _rbboxes = (
    #             det_rbboxes[:, :8] * scale_factor if rescale else det_rbboxes)
    #         _rects = rbboxes2rects(_rbboxes[:, :8])
    #         rrois = dbbox2roi([_rects])
    #         atten_feats = self.atten_rroi_extractor(
    #                     x[:self.atten_rroi_extractor.num_inputs], rrois)
    #         atten_maps = self.atten_head(atten_feats)
    #         base_maps = self.blender.crop_segm(basis_pred, rrois)
    #         mask_pred = self.blender.merge_bases(base_maps, atten_maps)
    #         segm_result = self.blender.get_rotate_seg_masks(mask_pred, 
    #                 _rects, det_labels, self.test_cfg.rcnn, ori_shape,
    #                 scale_factor, rescale)
    #     return segm_result

    def simple_test_rotate_mask(self, x, img_meta, 
            det_rbboxes, det_labels, rescale=False):
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_rbboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.blender.num_classes - 1)]
        else:
            # basis pred 
            basis_pred = self.basis_head.forward(x)
            # atten pred
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _rbboxes = (
                det_rbboxes[:, :8] * scale_factor if rescale else det_rbboxes)
            _rects = rbboxes2rects(_rbboxes[:, :8])
            rrois = dbbox2roi([_rects])
            atten_feats = self.atten_rroi_extractor(
                        x[:self.atten_rroi_extractor.num_inputs], rrois)
            atten_maps = self.atten_head(atten_feats)
            segm_result = self.blender.get_seg_masks(basis_pred, atten_maps, 
                rrois[:,1:], det_labels, self.test_cfg.rcnn, 
                ori_shape, scale_factor, rescale)
        return segm_result

    def aug_test(self, imgs, img_metas, proposals=None, rescale=None):
        print('aug_test is not support')