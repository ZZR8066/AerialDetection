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
    delta2bbox_8points
from ..builder import build_loss
from ..registry import HEADS

from .fused_rbbox_head import FusedFCBBoxHeadRbbox

@HEADS.register_module
class FusedVertexFCBBoxHeadRbbox(FusedFCBBoxHeadRbbox):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(FusedVertexFCBBoxHeadRbbox, self).__init__(
            num_fcs=2, 
            fc_out_channels=1024,
            *args,
            **kwargs)
        if self.with_reg:
            out_dim_reg = (8 if self.reg_class_agnostic else 8 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def get_target_rbbox(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        """
        obb target obb
        :param sampling_results:
        :param gt_bboxes:
        :param gt_labels:
        :param rcnn_train_cfg:
        :return:
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        # pos_proposals = choose_best_Rroi_batch(pos_proposals)
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = rbbox_target_vertex(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    def loss(self,
            cls_score,
            bbox_pred,
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
            losses['rbbox_loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0))
        return losses
    
    def get_det_rbboxes(self,
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=False,
                    cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            dbboxes = delta2bbox_8points(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape).reshape(-1, bbox_pred.shape[-1])
        else:
            dbboxes = rois[:, 1:]

        if rescale:
            dbboxes /= scale_factor

        c_device = dbboxes.device

        det_bboxes, det_labels = multiclass_nms_polygons(dbboxes, scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)

        return det_bboxes, det_labels