import torch
import numpy as np
import mmcv
import cv2

def mask_target_hbb(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets

def rbbox2bbox(rbbox):
    x,y,w,h,theta = rbbox
    rect = ((x,y),(w,h),theta*180/np.pi)
    polygon = cv2.boxPoints(rect)
    x1 = polygon[:,0].min()
    x2 = polygon[:,0].max()
    y1 = polygon[:,1].min()
    y2 = polygon[:,1].max()
    return x1, y1, x2, y2
    
def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            # import pdb
            # pdb.set_trace()
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            rows, cols = gt_mask.shape
            rbbox = proposals_np[i, :]
            x1, y1, x2, y2 = rbbox2bbox(rbbox)
            x1 = int(x1.clip(0,cols - 2))
            x2 = int(x2.clip(0,cols - 2))
            y1 = int(y1.clip(0,rows - 2))
            y2 = int(y2.clip(0,rows - 2))
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   (mask_size, mask_size))
            mask_targets.append(target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)
    else:
        mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
    return mask_targets
