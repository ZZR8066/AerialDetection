import torch
import numpy as np
import mmcv
import cv2

def get_mask_size(proposals_list, base_size, ratio_max):
    ratio = 1.0
    for proposals in proposals_list:
        ratios = proposals[:, 2] / proposals[:, 3]
        assert ratios.min() >= 1.0
        ratio = max(ratio, ratios.ceil().max())
    ratio = float(min(ratio_max, ratio))
    pool_h = int(base_size)
    pool_w = int(ratio * base_size)
    return pool_w, pool_h
    
def crop_rotate_mask(gt_mask, x,y,w,h,theta):
    rows, cols = gt_mask.shape
    expand_layer = np.zeros((rows*2,cols*2),dtype='uint8')
    rows_start = int(rows / 2)
    cols_start = int(cols / 2)
    expand_layer[rows_start:rows_start+rows, cols_start:cols_start+cols]=gt_mask
    M = cv2.getRotationMatrix2D((x+cols_start,y+rows_start),theta*180/np.pi,1)
    dst = cv2.warpAffine(expand_layer,M,expand_layer.shape,borderValue=0)
    M = np.float32([[1.0,0,-x+w/2-cols_start],[0,1,-y+h/2-rows_start]])
    dst = cv2.warpAffine(dst,M,dst.shape,borderValue=0)
    dst = dst[:np.int(h), :np.int(w)]
    return dst
    
def mask_rotate_fix_target(pos_proposals_list, pos_assigned_gt_inds_list, 
                gt_masks_list, cfg, out_w, out_h):
    mask_w = out_w
    mask_h = out_h
    mask_rotate_fix_targets_list = []
    for i in range(len(gt_masks_list)):
        mask_rotate_fix_targets = []
        
        pos_assigned_gt_inds = pos_assigned_gt_inds_list[i]
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        
        pos_proposals = pos_proposals_list[i]
        proposals_np = pos_proposals.cpu().numpy()
        
        gt_masks = gt_masks_list[i]
        
        num_pos = proposals_np.shape[0]
        
        if num_pos > 0:
            for j in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[j]]
                bbox = proposals_np[j, :]
                x, y, w, h, theta = bbox
                w = np.maximum(w, 1)
                h = np.maximum(h, 1)
                dst = crop_rotate_mask(gt_mask, x, y, w, h, theta)
                target = mmcv.imresize(dst, (mask_w, mask_h))
                mask_rotate_fix_targets.append(target)
            mask_rotate_fix_targets = torch.from_numpy(np.stack(mask_rotate_fix_targets)).float().to(
                pos_proposals.device)
        else:
            mask_rotate_fix_targets = pos_proposals.new_zeros((0, mask_h, mask_w))
        mask_rotate_fix_targets_list.append(mask_rotate_fix_targets)

    return torch.cat(mask_rotate_fix_targets_list)