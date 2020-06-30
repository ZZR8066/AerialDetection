import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import HEADS
from mmdet.models import builder
from mmdet.ops.roi_align_rotated import roi_align_rotated_cuda
import mmcv
import cv2
import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
@HEADS.register_module
class Blender(nn.Module):
    def __init__(self,
                num_classes,
                base_size,
                rroi_extractor):
        super(Blender, self).__init__()
        self.num_classes = num_classes
        self.base_size = base_size
        self.rroi_extractor = builder.build_roi_extractor(rroi_extractor)
            
    def crop_segm(self, bases, rrois):
        return self.rroi_extractor(tuple([bases]),rrois)
        
    # def merge_bases(self, base_maps, atten_maps):
    #     atten_maps = F.interpolate(atten_maps, (self.base_size, self.base_size), mode='bilinear').softmax(dim=1)
    #     mask_pred = (base_maps * atten_maps).sum(dim=1)
    #     return mask_pred
    
    def merge_bases(self, base_maps, atten_maps):
        atten_maps = F.interpolate(atten_maps, (base_maps.shape[-2], base_maps.shape[-1]), \
            mode='bilinear').softmax(dim=1)
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

    def get_rotate_seg_masks(self, mask_pred, det_rects, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        rects  = det_rects.cpu().numpy()[:, :5]
        labels = det_labels.cpu().numpy()

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(rects.shape[0]):
            rects[i, :4] = rects[i, :4] / scale_factor
            x,y,w,h,theta = rects[i,:]
            theta = -1*theta*180/np.pi
            w = int(np.maximum(w, 1))
            h = int(np.maximum(h, 1))
            label = labels[i]
            
            mask_pred_ = mask_pred[i, :, :]
            
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            expand_layer = np.zeros((img_h*2,img_w*2),dtype='uint8')
            w_start = int(img_w / 2)
            h_start = int(img_h / 2)
            
            rbbox_mask = mmcv.imresize(mask_pred_, (w, h))
            rbbox_mask = (rbbox_mask > rcnn_test_cfg.mask_thr_binary).astype(np.uint8)

            expand_layer[:h, :w] = rbbox_mask
            M = np.float32([[1,0,x - w/2 + w_start], [0, 1, y - h/2 + h_start]])
            expand_layer = cv2.warpAffine(expand_layer,M,expand_layer.shape[::-1], borderValue=0)
            M = cv2.getRotationMatrix2D((x+w_start,y+h_start),theta,1)
            expand_layer = cv2.warpAffine(expand_layer,M,expand_layer.shape[::-1],borderValue=0)

            im_mask = expand_layer[h_start:h_start+img_h, w_start:w_start+img_w]
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label].append(rle)

        return cls_segms

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy()
        
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            mask_pred_ = mask_pred[i, :, :]
            
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label].append(rle)

        return cls_segms

    # def get_seg_masks(self, basis_pred, atten_maps, det_rects, det_labels, rcnn_test_cfg,
    #                   ori_shape, scale_factor, rescale):

    #     cls_segms = [[] for _ in range(self.num_classes - 1)]
    #     rects  = det_rects
    #     labels = det_labels

    #     if rescale:
    #         img_h, img_w = ori_shape[:2]
    #     else:
    #         img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
    #         img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
    #         scale_factor = 1.0
        
    #     rects[:, :4] = rects[:, :4] / scale_factor
    #     # basis_pred = F.interpolate(basis_pred, (img_h, img_w), mode="bilinear", align_corners=False)
    #     feat_h = basis_pred.shape[-2]
    #     feat_w = basis_pred.shape[-1]
    #     scale_x = feat_w / img_w
    #     scale_y = feat_h / img_h
    #     rects[:,0] *= scale_x
    #     rects[:,1] *= scale_y
    #     rects[:,2] *= scale_x
    #     rects[:,3] *= scale_y
    #     atten_maps = paste_rotate_masks_in_image(atten_maps, rects, feat_h, feat_w)
    #     mask_pred = self.merge_bases(basis_pred, atten_maps)
    #     mask_pred = mask_pred.sigmoid()
    #     mask_pred[~(paste_rotate_masks_in_image(atten_maps.new_ones(atten_maps.size())\
    #         , rects, feat_h, feat_w).to(torch.bool)[:,0,:,:])] = 0.0
        
    #     for i in range(mask_pred.shape[0]):
    #         label = labels[i]
    #         im_mask = mask_pred[i,:,:].cpu().numpy()
    #         im_mask = mmcv.imresize(im_mask,(img_w, img_h))
    #         im_mask = (im_mask > rcnn_test_cfg.mask_thr_binary).astype('uint8')

    #         rle = mask_util.encode(
    #             np.array(im_mask[:, :, np.newaxis], order='F'))[0]
    #         cls_segms[label].append(rle)
    #     return cls_segms

import cv2
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F

BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 512 ** 3  # 1 GB memory limit

def _do_paste_rotate_mask(masks, rboxes, img_h, img_w):
    """
    Args:
        masks: None, None, H, W
        rboxes: N, 5
        img_h, img_w (int):
    Returns:
        a mask of shape (N, img_h, img_w)
    """
    N = masks.shape[0]
    device = masks.device
    x0_int, y0_int = 0, 0
    x1_int, y1_int = img_w, img_h
    xc, yc, w, h, theta = torch.split(rboxes, 1, dim=-1)  # each is Nx1

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    # center
    img_y = (img_y - yc)
    img_x = (img_x - xc)
    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    # rotate
    gx_rotate = ((gx.reshape(N,-1)*torch.cos(theta) + gy.reshape(N,-1)* \
        torch.sin(theta)) / w * 2).reshape(gx.size())
    gy_rotate = ((gy.reshape(N,-1)*torch.cos(theta) - gx.reshape(N,-1)* \
        torch.sin(theta)) / h * 2).reshape(gy.size())
    grid = torch.stack([gx_rotate, gy_rotate], dim=3)

    img_mask = F.grid_sample(masks.to(dtype=torch.float32), grid, align_corners=False)
    return img_mask

def paste_rotate_masks_in_image(masks, rboxes, img_h, img_w):
    
    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = masks.shape[0]
    K = masks.shape[1]

    device = rboxes.device
    assert len(rboxes) == N, boxes.shape

    # GPU benefits from parallelism for larger chunks, but may have memory issue
    # int(img_h) because shape may be tensors in tracing
    num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
    assert (
        num_chunks <= N
    ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, K, img_h, img_w, device=device, dtype=masks.dtype
    )
    for inds in chunks:
        masks_chunk = _do_paste_rotate_mask(
            masks[inds, :, :, :], rboxes[inds], img_h, img_w
        )

        img_masks[(inds,)] = masks_chunk
    return img_masks