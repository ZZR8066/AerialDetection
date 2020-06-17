import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import HEADS
from mmdet.models import builder
from mmdet.ops.roi_align_rotated import roi_align_rotated_cuda
    
@HEADS.register_module
class Blender(nn.Module):
    def __init__(self,
                base_size,
                rroi_extractor):
        super(Blender, self).__init__()
        self.base_size = base_size
        self.rroi_extractor = builder.build_roi_extractor(rroi_extractor)
            
    def crop_segm(self, bases, rrois):
        return self.rroi_extractor(tuple([bases]),rrois)
        
    def merge_bases(self, base_maps, atten_maps):
        atten_maps = F.interpolate(atten_maps, (self.base_size, self.base_size), mode='bilinear').softmax(dim=1)
        mask_pred = (base_maps * atten_maps).sum(dim=1)
        return mask_pred
    
    def get_target(self, sampling_results, gt_masks_list):
        pos_proposals_list = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds_list = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_size = self.base_size
        mask_targets_list = []
        for pos_proposals, pos_assigned_gt_inds, gt_masks \
            in zip(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list):
            num_pos = pos_proposals.size(0)
            mask_targets = []
            if num_pos > 0:
                for i in range(num_pos):
                    gt_mask = gt_masks[pos_assigned_gt_inds[i]]
                    gt_mask = torch.from_numpy(gt_mask[np.newaxis, \
                        np.newaxis, :, :]).float().to(pos_proposals.device)
                    x,y,w,h,theta = pos_proposals[i, :]
                    roi = torch.tensor([[0.0, x, y, w, h, theta]]).to(pos_proposals.device)
                    target = pos_proposals.new_zeros(1, 1, mask_size, mask_size)
                    roi_align_rotated_cuda.forward(gt_mask, roi, mask_size, mask_size, 1.0, 2, target)
                    mask_targets.append(target.squeeze())
                    # mask_targets.append(torch.where(target>0.5,target.new_full(target.shape, 1),target.new_full(target.shape, 0)))
            else:
                mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
            mask_targets_list.append(torch.stack(mask_targets))
        return torch.cat(mask_targets_list)

    def loss(self, pred, target):
        loss = dict()
        loss['loss_mask'] = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')[None]
        return loss