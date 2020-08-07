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

from .fused_rbbox_head import FusedFCBBoxHeadRbbox

@HEADS.register_module
class FusedAPEFCBBoxHeadRbbox(FusedFCBBoxHeadRbbox):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(FusedAPEFCBBoxHeadRbbox, self).__init__(
            num_fcs=2, 
            fc_out_channels=1024,
            *args,
            **kwargs)
        if self.with_reg:
            out_dim_reg = (8 if self.reg_class_agnostic else 8 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)
    
    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # apply sigmoid to theta part
        bbox_pred_size = bbox_pred.size()
        bbox_pred = bbox_pred.view(bbox_pred_size[0], -1, 8)
        bbox_pred_xy = bbox_pred[:, :, :4]
        bbox_pred_theta = torch.sigmoid(bbox_pred[:, :, 4:]) * 2 - 1
        bbox_pred = torch.cat((bbox_pred_xy, bbox_pred_theta), dim=2).view(bbox_pred_size)

        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_obbs, gt_labels,
                   rcnn_train_cfg):
        """
        obb target hbb
        :param sampling_results:
        :param gt_masks:
        :param gt_labels:
        :param rcnn_train_cfg:
        :param mod: 'normal' or 'best_match', 'best_match' is used for RoI Transformer
        :return:
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        # pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        # TODO: first get indexs of pos_gt_bboxes, then index from gt_bboxes
        # TODO: refactor it, direct use the gt_rbboxes instead of gt_masks
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds  for res in sampling_results
        ]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target_rbbox_APE(
            pos_proposals,
            neg_proposals,
            pos_assigned_gt_inds,
            gt_obbs,
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
    
    # def loss(self,
    #          cls_score,
    #          bbox_pred,
    #          sampling_results,
    #          gt_obbs_torch,
    #          labels,
    #          label_weights,
    #          bbox_targets,
    #          bbox_weights,
    #          reduce=True):
    #     losses = dict()
    #     if cls_score is not None:
    #         losses['rbbox_loss_cls'] = self.loss_cls(
    #             cls_score, labels, label_weights, reduce=reduce)
    #         losses['rbbox_acc'] = accuracy(cls_score, labels)
    #     if bbox_pred is not None:
    #         pos_inds = labels > 0
    #         if self.reg_class_agnostic:
    #             pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 8)[pos_inds]
    #         else:
    #             pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
    #                                            8)[pos_inds, labels[pos_inds]]
    #         with torch.no_grad():
    #             pos_proposals = [res.pos_bboxes for res in sampling_results]
    #             pos_assigned_gt_inds = [res.pos_assigned_gt_inds  for res in sampling_results]
    #             gt_obbs = [gt[inds.cpu().numpy()] \
    #                 for inds, gt in zip(pos_assigned_gt_inds, gt_obbs_torch)]
    #             gt_obbs = torch.cat(gt_obbs, 0)
    #             pos_proposals = torch.cat(pos_proposals, 0)
    #             pos_bbox = delta2bbox_APE(pos_proposals, pos_bbox_pred, self.target_means,
    #                         self.target_stds)
    #         losses['rbbox_loss_bbox'] = self.loss_bbox(
    #             pos_bbox_pred,
    #             bbox_targets[pos_inds],
    #             pos_bbox,
    #             gt_obbs,
    #             bbox_weights[pos_inds])
    #     return losses

    def refine_rbboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 5) or (n*bs, 5*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class_rbbox(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    def regress_by_class_rbbox(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 5) or (n, 6)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 5*(#class+1)) or (n, 5)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        # import pdb
        # pdb.set_trace()
        assert rois.size(1) == 4

        if not self.reg_class_agnostic:
            import pdb
            pdb.set_trace()
        assert bbox_pred.size(1) == 8

        new_rois = delta2bbox_APE(rois, bbox_pred, self.target_means,
           self.target_stds, img_meta['img_shape'])

        return new_rois

    def get_det_bboxes(self,
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
            dbboxes = delta2bbox_APE(rois[:, 1:], bbox_pred, self.target_means,
                                        self.target_stds, img_shape)
        else:
            # bboxes = rois[:, 1:]
            dbboxes = roi2droi(rois)[:, 1:]
            # dbboxes = rrois[:, 1:]
            # TODO: add clip here

        if rescale:
            # bboxes /= scale_factor
            # dbboxes[:, :4] /= scale_factor
            dbboxes[:, 0::5] /= scale_factor
            dbboxes[:, 1::5] /= scale_factor
            dbboxes[:, 2::5] /= scale_factor
            dbboxes[:, 3::5] /= scale_factor
        if cfg is None:
            return dbboxes, scores
        else:
            c_device = dbboxes.device

            det_bboxes, det_labels = multiclass_nms_rbbox(dbboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            # det_bboxes = torch.from_numpy(det_bboxes).to(c_device)
            # det_labels = torch.from_numpy(det_labels).to(c_device)
            return det_bboxes, det_labels