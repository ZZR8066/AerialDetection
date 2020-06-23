import torch
import numpy as np
import mmcv
import cv2

def mask_rotate_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_rotate_targets = map(mask_rotate_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_rotate_targets = torch.cat(list(mask_rotate_targets))
    return mask_rotate_targets

def crop_rotate_mask(gt_mask, x,y,w,h,theta):
    rows, cols = gt_mask.shape
    expand_layer = np.zeros((rows*2,cols*2),dtype='uint8')
    rows_start = int(rows / 2)
    cols_start = int(cols / 2)
    expand_layer[rows_start:rows_start+rows, cols_start:cols_start+cols]=gt_mask
    M = cv2.getRotationMatrix2D((x+cols_start,y+rows_start),theta*180/np.pi,1)
    dst = cv2.warpAffine(expand_layer,M,expand_layer.shape[::-1],borderValue=0)
    M = np.float32([[1.0,0,-x+w/2-cols_start],[0,1,-y+h/2-rows_start]])
    dst = cv2.warpAffine(dst,M,dst.shape[::-1],borderValue=0)
    dst = dst[:np.int(h), :np.int(w)]
    return dst

def mask_rotate_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_rotate_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            # import pdb
            # pdb.set_trace()
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :]
            x, y, w, h, theta = bbox
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)
            dst = crop_rotate_mask(gt_mask, x, y, w, h, theta)
            # mask is uint8 both before and after resizing
            target = mmcv.imresize(dst, (mask_size, mask_size))
            mask_rotate_targets.append(target)
        mask_rotate_targets = torch.from_numpy(np.stack(mask_rotate_targets)).float().to(
            pos_proposals.device)
    else:
        mask_rotate_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
    return mask_rotate_targets
    
