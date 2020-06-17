from .. import builder
from .fcos import FCOS
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import bbox2roi
import torch

@DETECTORS.register_module
class BlendMask(FCOS):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(BlendMask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

        self.mask_roi_extractor = builder.build_roi_extractor(mask_roi_extractor)
        self.mask_head = builder.build_head(mask_head)
        self.mask_head.init_weights()
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        # print('in single_stage')
        # import pdb
        # pdb.set_trace()
        losses = dict()
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        bboxes_loss = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(bboxes_loss)
        # mask head forward and loss
        bottom_feat, atten_feat = self.mask_head(x)
        pos_rois = bbox2roi(gt_bboxes)
        bottom_pred = self.mask_roi_extractor([bottom_feat], pos_rois)
        atten_pred = self.mask_head.get_correspond_atten(self.bbox_head.regress_ranges, 
                            self.bbox_head.strides, atten_feat, gt_bboxes)
        mask_pred = self.mask_head.blender(bottom_pred, atten_pred)
        mask_target = self.mask_head.get_mask_target(gt_bboxes, gt_masks)
        mask_loss = self.mask_head.loss(mask_pred.unsqueeze(1), mask_target, torch.cat(gt_labels))
        losses.update(mask_loss)
        return losses
