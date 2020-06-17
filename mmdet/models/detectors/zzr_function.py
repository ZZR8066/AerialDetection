import cv2
import torch
import numpy as np


def test_rroi2roi(rrois,rois):
    mask = np.zeros([800,800], dtype=np.uint8)
    for rroi, roi in zip(rrois, rois):
        x,y,w,h,theta = rroi[1:]
        rect = ((float(x), float(y)),
                (float(w), float(h)),
                    float(theta) * 180 / np.pi)
        polygon = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(mask,[polygon],0,255,1)
        x1,y1,x2,y2 = roi[1:]
        cv2.rectangle(mask, (int(x1),int(y1)),
            (int(x2),int(y2)), 255, 2)
    cv2.imwrite("0.png",mask)

def rrois2rois(rrois):
    rois = torch.empty(rrois.shape[0],5).cuda(rrois.device).type(rrois.dtype)
    if rois.shape[0] == 0:
        return rois
    rois[:, 0] = rrois[:, 0]
    for i in range(rrois.shape[0]):
        x,y,w,h,theta = rrois[i,1:]
        rect = ((x.item(), y.item()), 
                (w.item(), h.item()),
                 theta.item() * 180 / np.pi)
        polygon = cv2.boxPoints(rect)
        x1 = polygon[:,0].min()
        x2 = polygon[:,0].max()
        y1 = polygon[:,1].min()
        y2 = polygon[:,1].max()
        rois[i][1] = float(x1)
        rois[i][2] = float(y1)
        rois[i][3] = float(x2)
        rois[i][4] = float(y2)
    return rois
    
def draw_gt_obboxes(gt_masks, gt_obboxes):
    '''
        绘制 旋转框 用于检验
    '''
    mask = np.zeros(gt_masks[0].shape, dtype='uint8')
    for i, gt_mask in enumerate(gt_masks):
        mask = mask + gt_mask*255
        gt_obbox = gt_obboxes[i]
        x,y,w,h,theta = gt_obbox
        rect = ((x,y),(w,h),theta*180/np.pi)
        box = cv2.boxPoints(rect)
        box = box.astype('int')
        cv2.drawContours(mask,[box],0,255,2)
    cv2.imwrite("0.png",mask)

def det_rbboxes2det_bboxes(det_rbboxes):
    det_bboxes = torch.empty(len(det_rbboxes),5).cuda(det_rbboxes.device).type(torch.float32)
    if len(det_rbboxes) == 0:
        return det_bboxes
    det_bboxes[:,-1] = det_rbboxes[:,-1]
    for i in range(len(det_rbboxes)):
        polygon = det_rbboxes[i][:8]
        polygon = polygon.reshape(4,2)
        x1 = polygon[:,0].min()
        x2 = polygon[:,0].max()
        y1 = polygon[:,1].min()
        y2 = polygon[:,1].max()
        det_bboxes[i][0] = x1
        det_bboxes[i][1] = y1
        det_bboxes[i][2] = x2
        det_bboxes[i][3] = y2
    return det_bboxes
      
      
def rbboxes2rects(rbboxes):
    '''
        四点坐标转成旋转框形式 x,y,w,h,theta
    '''
    rects = rbboxes.new_zeros(rbboxes.shape[0],5).float()
    rbboxes = rbboxes.cpu().numpy()
    for i,rbbox in enumerate(rbboxes):
        rect = cv2.minAreaRect(rbbox.astype(np.int64).reshape(-1,1,2))
        xc = rect[0][0]
        yc = rect[0][1]
        w  = rect[1][0]
        h  = rect[1][1]
        theta = rect[2] / 180 * np.pi
        if w > h:
            rects[i][0] = xc
            rects[i][1] = yc
            rects[i][2] = w
            rects[i][3] = h
            rects[i][4] = theta
        else:
            rects[i][0] = xc
            rects[i][1] = yc
            rects[i][2] = h
            rects[i][3] = w
            rects[i][4] = (theta + np.pi / 2)
    return rects
    
def concatenate_points(c):
    '''
        级联 list c
    '''
    t = c[0]
    for i in range(1,len(c)):
        t = np.concatenate((t,c[i]),axis=0)
    return t
    
def mask2obb_mask(gt_bboxes_list, gt_masks_list, enlarge_pixel=2):
    '''
        mask 转成旋转框mask标注
    '''
    masks_list = []
    for i in range(len(gt_masks_list)):
        gt_masks = gt_masks_list[i]
        gt_bboxes = gt_bboxes_list[i]
        masks = np.zeros(gt_masks.shape, dtype='uint8')
        for j in range(len(gt_masks)):
            gt_mask  = gt_masks[j]
            gt_bbox  = gt_bboxes[j]
            contours, hierarchy = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours == []: # 没找到边缘线 用水平框
                x1,y1,x2,y2 = gt_bbox
                bbox = np.zeros([4,2], dtype = 'int')
                bbox[0,0] = x1
                bbox[0,1] = y1
                bbox[1,0] = x2
                bbox[1,1] = y1
                bbox[2,0] = x2
                bbox[2,1] = y2
                bbox[3,0] = x1
                bbox[3,1] = y2
                mask_bbox = np.zeros(gt_mask.shape, dtype = 'uint8')
                masks[j,:,:] = cv2.fillPoly(mask_bbox,[bbox],1)
            else: # 找到边缘线    
                contour = concatenate_points(contours) # 级联所有的点
                rect = cv2.minAreaRect(contour)
                rect = (rect[0], (rect[1][0] + enlarge_pixel, rect[1][1] + enlarge_pixel), rect[2])
                poly = cv2.boxPoints(rect)
                # 检查poly是否有重复点
                is_repeat = False
                for m in range(poly.shape[0] - 1):
                    for n in range(m+1, poly.shape[0]):
                        f = poly[m] == poly[n]
                        if f.all():
                            is_repeat = True
                if is_repeat: # 若有重复点采用水平框标注
                    x1,y1,x2,y2 = gt_bbox
                    bbox = np.zeros([4,2], dtype = 'int')
                    bbox[0,0] = x1
                    bbox[0,1] = y1
                    bbox[1,0] = x2
                    bbox[1,1] = y1
                    bbox[2,0] = x2
                    bbox[2,1] = y2
                    bbox[3,0] = x1
                    bbox[3,1] = y2
                    mask_bbox = np.zeros(gt_mask.shape, dtype = 'uint8')
                    masks[j,:,:] = cv2.fillPoly(mask_bbox,[bbox],1)
                else:         # 若没有重复采用旋转框
                    mask_bbox = np.zeros(gt_mask.shape, dtype = 'uint8')
                    masks[j,:,:] = cv2.fillPoly(mask_bbox,[np.int0(poly)],1)
        masks_list.append(masks)
    return masks_list

def tran2mix_mask(gt_bboxes_list, gt_masks_list, gt_labels_list, 
        tran_labels=[1,2,5,6,7,8,9,10,12,13,15]):
    '''
        'Small_Vehicle',  1
        'Large_Vehicle',  2
        'plane',          3
        'storage_tank',   4
        'ship',           5
        'Swimming_pool',  6
        'Harbor',         7
        'tennis_court',   8
        'Ground_Track_Field',  9
        'Soccer_ball_field',   10
        'baseball_diamond',    11
        'Bridge',              12
        'basketball_court',    13
        'Roundabout',          14
        'Helicopter'           15
        
        将对应类别转成回归旋转框的mask
    '''
    mix_masks_list = []
    for i in range(len(gt_masks_list)):
        gt_masks = gt_masks_list[i]
        gt_bboxes = gt_bboxes_list[i]
        gt_labels = gt_labels_list[i]
        mix_masks = np.zeros(gt_masks.shape, dtype='uint8')
        for j in range(len(gt_masks)):
            gt_mask  = gt_masks[j]
            gt_bbox  = gt_bboxes[j]
            gt_label = gt_labels[j]
            if gt_label in tran_labels: # 转成旋转框
                contours, hierarchy = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contours == []: # 没找到边缘线 用水平框
                    x1,y1,x2,y2 = gt_bbox
                    bbox = np.zeros([4,2], dtype = 'int')
                    bbox[0,0] = x1
                    bbox[0,1] = y1
                    bbox[1,0] = x2
                    bbox[1,1] = y1
                    bbox[2,0] = x2
                    bbox[2,1] = y2
                    bbox[3,0] = x1
                    bbox[3,1] = y2
                    mask_bbox = np.zeros(gt_mask.shape, dtype = 'uint8')
                    mix_masks[j,:,:] = cv2.fillPoly(mask_bbox,[bbox],1)
                else:              # 找到边缘线    
                    contour = concatenate_points(contours) # 级联所有的点
                    rect = cv2.minAreaRect(contour)
                    poly = cv2.boxPoints(rect)
                    # 检查poly是否有重复点
                    is_repeat = False
                    for m in range(poly.shape[0] - 1):
                        for n in range(m+1, poly.shape[0]):
                            f = poly[m] == poly[n]
                            if f.all():
                                is_repeat = True
                    if is_repeat: # 若有重复点采用水平框标注
                        x1,y1,x2,y2 = gt_bbox
                        bbox = np.zeros([4,2], dtype = 'int')
                        bbox[0,0] = x1
                        bbox[0,1] = y1
                        bbox[1,0] = x2
                        bbox[1,1] = y1
                        bbox[2,0] = x2
                        bbox[2,1] = y2
                        bbox[3,0] = x1
                        bbox[3,1] = y2
                        mask_bbox = np.zeros(gt_mask.shape, dtype = 'uint8')
                        mix_masks[j,:,:] = cv2.fillPoly(mask_bbox,[bbox],1)
                    else:         # 若没有重复采用旋转框
                        mask_bbox = np.zeros(gt_mask.shape, dtype = 'uint8')
                        mix_masks[j,:,:] = cv2.fillPoly(mask_bbox,[np.int0(poly)],1)
            else:                       # 用水平框
                x1,y1,x2,y2 = gt_bbox
                bbox = np.zeros([4,2], dtype = 'int')
                bbox[0,0] = x1
                bbox[0,1] = y1
                bbox[1,0] = x2
                bbox[1,1] = y1
                bbox[2,0] = x2
                bbox[2,1] = y2
                bbox[3,0] = x1
                bbox[3,1] = y2
                mask_bbox = np.zeros(gt_mask.shape, dtype = 'uint8')
                mix_masks[j,:,:] = cv2.fillPoly(mask_bbox,[bbox],1)
        mix_masks_list.append(mix_masks)
    return mix_masks_list

def mask2minArea(mask_pred_list, gt_masks_list, sampling_results):
    '''
        mask --> rects
    '''
    bboxes_list = [res.pos_bboxes for res in sampling_results]
    if isinstance(mask_pred_list, torch.Tensor):
        mask_pred_list = mask_pred_list.sigmoid().data.cpu().numpy()
    assert isinstance(mask_pred_list, np.ndarray)
    mask_pred_list = mask_pred_list.astype(np.float32)
    rects_list = []
    start = 0
    stop  = 0
    for i in range(len(bboxes_list)):
        bboxes = bboxes_list[i]
        gt_masks = gt_masks_list[i]
        stop  = stop + len(bboxes)
        rects  = bboxes.new_zeros(len(bboxes),5)
        mask_pred = mask_pred_list[start:stop]
        bboxes = bboxes.cpu().numpy()[:,:4]
        for j in range(bboxes.shape[0]):
            bbox  = bboxes[j, :]
            w = int(max(bbox[2] - bbox[0] + 1, 1))
            h = int(max(bbox[3] - bbox[1] + 1, 1))
            gt_mask = gt_masks[j]
            mask_pred_ = mask_pred[j, :, :]
            bbox_mask  = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask  = (bbox_mask > 0.5).astype(np.uint8)
            im_mask = np.zeros(gt_mask.shape, dtype=np.uint8)
            im_mask[int(bbox[1]):int(bbox[1] + h), int(bbox[0]):int(bbox[0] + w)] = bbox_mask
            contours, _ = cv2.findContours(im_mask*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours == []: 
                x1,y1,x2,y2 = bbox
                rects[j,0] = (x1 + x2) / 2
                rects[j,1] = (y1 + y2) / 2
                w = x2 - x1  + 1.0
                h = y2 - y1  + 1.0
                if w > h:
                    rects[j,2] =  w
                    rects[j,3] =  h
                    rects[j,4] =  0.0
                else:
                    rects[j,2] =  h
                    rects[j,3] =  w
                    rects[j,4] =  np.pi / 2
            else: 
                c = sorted(contours, key=cv2.contourArea, reverse=True)
                c = concatenate_points(c) 
                rec = cv2.minAreaRect(c)
                rects[j,0] = rec[0][0]
                rects[j,1] = rec[0][1]
                w = rec[1][0]
                h = rec[1][1]
                if w > h:
                    rects[j,2] = w
                    rects[j,3] = h
                    rects[j,4] = rec[2] / 180.0 * np.pi
                else:
                    rects[j,2] = h
                    rects[j,3] = w
                    rects[j,4] = (rec[2]+90) / 180.0 * np.pi
        rects_list.append(rects)
        start = stop
    return rects_list

def crop_rotate_mask(gt_mask, x, y, w, h, theta):
    rows, cols = gt_mask.shape
    expand_layer = np.zeros((rows*2,cols*2),dtype='uint8')
    rows_start = int(rows / 2)
    cols_start = int(cols / 2)
    expand_layer[rows_start:rows_start+rows, cols_start:cols_start+cols]=gt_mask
    w = max(w,1)
    h = max(h,1)
    w = min(w,cols - 1)
    h = min(h,rows - 1)
    M = cv2.getRotationMatrix2D((x+cols_start,y+rows_start),theta*180/np.pi,1)
    dst = cv2.warpAffine(expand_layer,M,expand_layer.shape[::-1],borderValue=0)
    M = np.float32([[1.0,0,-x+w/2-cols_start],[0,1,-y+h/2-rows_start]])
    dst = cv2.warpAffine(dst,M,dst.shape[::-1],borderValue=0)
    dst = dst[:np.int(h), :np.int(w)]
    return dst

def trans2hbb(proposals_list):
    final_list = []
    for proposals in proposals_list:
        hbbs = torch.empty(proposals.shape[0],4).cuda(proposals.device).type(proposals.dtype)
        if proposals.shape[0] == 0:
            final_list.append(hbbs)
        else:
            for i in range(proposals.shape[0]):
                x,y,w,h,theta = proposals[i,:]
                rect = ((x.item(), y.item()), 
                        (w.item(), h.item()),
                         theta.item() * 180 / np.pi)
                polygon = cv2.boxPoints(rect)
                x1 = polygon[:,0].min()
                x2 = polygon[:,0].max()
                y1 = polygon[:,1].min()
                y2 = polygon[:,1].max()
                hbbs[i][0] = float(x1)
                hbbs[i][1] = float(y1)
                hbbs[i][2] = float(x2)
                hbbs[i][3] = float(y2)
            final_list.append(hbbs)
    return final_list