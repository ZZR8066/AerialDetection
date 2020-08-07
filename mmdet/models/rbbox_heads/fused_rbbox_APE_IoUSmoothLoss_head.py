import torch
import torch.nn as nn
import torch.nn.functional as F

from .rbbox_head import BBoxHeadRbbox
from ..registry import HEADS
from ..utils import ConvModule

from mmdet.core import delta2dbbox, multiclass_nms_rbbox, \
    bbox_target_rbbox, accuracy, rbbox_target_rbbox,\
    choose_best_Rroi_batch, delta2dbbox_v2, \
    Pesudomulticlass_nms_rbbox, delta2dbbox_v3, \
    hbb2obb_v2, rbbox_target_vertex, multiclass_nms_polygons, \
    delta2bbox_8points, bbox_target_rbbox_APE, delta2bbox_APE, roi2droi
from ..builder import build_loss
from ..registry import HEADS
from mmdet.core.bbox.geometry import rbbox_overlaps_cy_warp
from .fused_rbbox_APE_head import FusedAPEFCBBoxHeadRbbox

@HEADS.register_module
class FusedAPEFCBBoxHeadRbboxIoUSmoothLoss(FusedAPEFCBBoxHeadRbbox):
    # pass
    def loss(self,
             cls_score,
             bbox_pred,
             proposals,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduce=True):
        losses = dict()
        if cls_score is not None:
            losses['rbbox_loss_cls'] = self.loss_cls(
                cls_score, labels, label_weights, reduce=reduce)
            losses['rbbox_acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 8)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               8)[pos_inds, labels[pos_inds]]
            with torch.no_grad():
                pred_bbox = delta2bbox_APE(proposals, bbox_pred, self.target_means,
                            self.target_stds)
                gt_obbs = delta2bbox_APE(proposals, bbox_targets, self.target_means,
                            self.target_stds)
                            
                # overlaps = rbbox_overlaps_cy_warp(pred_bbox.cpu().numpy(), gt_obbs)
                # overlaps = overlaps.diag().reshape(-1,1).float()
                # overlaps = overlaps.clamp(1e-4, 1.0)
                # iou_factor = -1*torch.log(overlaps)


            losses['rbbox_loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                pred_bbox[pos_inds],
                gt_obbs[pos_inds],
                # iou_factor[pos_inds])
                bbox_weights[pos_inds])
                # bbox_weights[pos_inds,:1].reshape(-1,1))
        return losses