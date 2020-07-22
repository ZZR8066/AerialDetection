import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmcv.image import imread, imwrite
from mmdet.core import tensor2imgs, get_classes
import os

cls_names = [
	'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', \
	'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',  \
	'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

def DotaResult2Submit(file_name, result, save_dir):
    classes_all=[cls_names[var] for var in range(len(cls_names))]
    files=[save_dir + 'Task1_' + var + '.txt' for var in classes_all]
    files=[open(var,'a') for var in files]
    
    for i, cur_class_polygons in enumerate(result):
        if len(cur_class_polygons) == 0:
            continue
        else:
            f_class=files[i]
            for polygon in cur_class_polygons:
                score = polygon[-1]
                polygon = polygon[:8].reshape(4,2)
                f_class.write(os.path.basename(file_name).split('.')[0] \
                    + ' %f %f %f %f %f %f %f %f %f\n' % (score, \
                    polygon[0,0],polygon[0,1],polygon[1,0],polygon[1,1],\
                    polygon[2,0],polygon[2,1],polygon[3,0],polygon[3,1],))


def show_rmask_single(data,result, img_norm_cfg, class_names,
            score_thr=0.0, file_name='0.png'):
    
    bbox_result, segm_result, rbbox_result = result
    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        
        bboxes  = np.vstack(bbox_result)
        rbboxes = np.vstack(rbbox_result)
        rbboxes = rbboxes[:, :8]
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        
        rbbox_color = (0, 0, 255)
        text_color = (0, 255, 0)
        font_scale = 0.5
        
        for i in inds:
            imgs = tensor2imgs(img_tensor, **img_norm_cfg)
            img_show = imgs[0][:h, :w, :]
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            rbbox_int = rbboxes[i].astype(np.int32)
            rbbox_int = rbbox_int.reshape(4,2)
            cv2.drawContours(img_show,[rbbox_int],0,rbbox_color,2)
            path_str = 'test_out_img/' + str(i) + '.png'
            cv2 .imwrite(path_str, img_show)

def assembel_mask(results):
    assembel_masks = []
    for result in results:
        hbb_masks = result[1]
        obb_masks = result[-1]
        for i in range(len(hbb_masks)):
            for j in range(len(hbb_masks[i])):
                hbb_mask = hbb_masks[i][j]
                obb_mask = obb_masks[i][j]
                hbb_mask = maskUtils.decode(hbb_mask).astype(np.bool)
                obb_mask = maskUtils.decode(obb_mask).astype(np.bool)
                mask = hbb_mask + obb_mask
                mask = mask.astype(np.uint8)
                mask = maskUtils.encode(
                np.array(mask[:, :, np.newaxis], order='F'))[0]
                result[1][i][j] = mask
        assembel_masks.append(result[:2])
    return assembel_masks

def assembel_mask_V2(results):
    assembel_masks = []
    for result in results:
        hbb_masks  = result[1]
        obb_masks  = result[-1]
        obb_bboxes = result[-2]
        for i in range(len(hbb_masks)):
            for j in range(len(hbb_masks[i])):
                hbb_mask = hbb_masks[i][j]
                obb_mask = obb_masks[i][j]
                hbb_mask = maskUtils.decode(hbb_mask).astype(np.bool)
                obb_mask = maskUtils.decode(obb_mask).astype(np.bool)
                obb_bbox = obb_bboxes[i][j][:8]
                obb_bbox = obb_bbox.astype(np.int32)
                obb_bbox = obb_bbox.reshape(4,2)
                obb_bbox_mask = np.zeros(hbb_mask.shape[:2])
                cv2.fillPoly(obb_bbox_mask, [obb_bbox], 1)
                obb_bbox_mask = obb_bbox_mask.astype(np.bool)
                hbb_mask = np.multiply(obb_bbox_mask, hbb_mask).astype(np.bool)
                mask = hbb_mask + obb_mask
                mask = mask.astype(np.uint8)
                mask = maskUtils.encode(
                np.array(mask[:, :, np.newaxis], order='F'))[0]
                result[1][i][j] = mask
        assembel_masks.append(result[:2])
    return assembel_masks

def tran2obb_results(outputs):
    final_outs = []
    for output in outputs:
        output = [output[0], output[-1]]
        output = tuple(output)
        final_outs.append(output)
    return final_outs

def tran2hbb_results(outputs):
    final_outs = []
    for output in outputs:
        output = [output[0], output[1]]
        output = tuple(output)
        final_outs.append(output)
    return final_outs

def tran2mix_results(outputs, inds=[0,3,4,5,7,8,9,10,11,13,14]):
    final_results = []
    for output in outputs:
        output = list(output)
        for ind in inds:
            output[3][ind] = output[1][ind]
        output = [output[0], output[3]]
        output = tuple(output)
        final_results.append(output)
    return final_results

def show_all_box(data,result, img_norm_cfg, class_names,
            score_thr=0.3, file_name='0.png'):
    
    bbox_result, segm_result, rbbox_result = result
    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        
        bboxes  = np.vstack(bbox_result)
        rbboxes = np.vstack(rbbox_result)

        
        img = imread(img_show)
        
        scores = rbboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        rbboxes = rbboxes[inds, :8]
        
        rbbox_color = [(0, 0, 255),(0, 255, 0),(255, 0, 0),\
            (10, 100, 255), (1, 0, 255)]
        text_color = (0, 255, 0)
        font_scale = 0.5
        
        for rbbox, bbox in zip(rbboxes, bboxes):
            bbox_int = bbox.astype(np.int32)
            rbbox_int = rbbox.astype(np.int32)
            rbbox_int = rbbox_int.reshape(4,2)
            choice = np.random.choice(5, 1)
            cv2.drawContours(img,[rbbox_int],0,rbbox_color[choice[0]],3)
        
        cv2.imwrite('0.png',img)

    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        
        bboxes  = np.vstack(bbox_result)
        
        img = imread(img_show)
        
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        
        bbox_color = [(0, 0, 255),(0, 255, 0),(255, 0, 0),\
            (10, 100, 255), (1, 0, 255)]
        text_color = (0, 255, 0)
        font_scale = 0.5
        
        for bbox in bboxes:
            bbox_int = bbox.astype(np.int32)
            choice = np.random.choice(5, 1)
            cv2.rectangle(img, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), bbox_color[choice[0]], 2)
        
        cv2.imwrite('1.png',img)

def show_rmask(data,result, img_norm_cfg, class_names,
            score_thr=0.3, file_name='0.png'):
    
    bbox_result, segm_result, rbbox_result = result
    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        
        bboxes  = np.vstack(bbox_result)
        rbboxes = np.vstack(rbbox_result)

        # draw segmentation masks
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            for i in inds:
                color_mask = np.random.randint(
                    0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
        # draw rbbox
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        
        img = imread(img_show)
        
        scores = rbboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        rbboxes = rbboxes[inds, :8]
        labels = labels[inds]
        
        rbbox_color = (0, 0, 255)
        text_color = (0, 255, 0)
        font_scale = 0.5
        '''
        for rbbox, bbox, label in zip(rbboxes, bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            rbbox_int = rbbox.astype(np.int32)
            rbbox_int = rbbox_int.reshape(4,2)
            cv2.drawContours(img,[rbbox_int],0,rbbox_color,2)
            label_text = class_names[
                label] if class_names is not None else 'cls {}'.format(label)
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        '''
        cv2.imwrite(file_name,img)

def show_rbbox(data,result, img_norm_cfg, class_names,
            score_thr=0.3, file_name='0.png'):
    
    rbbox_result = result
    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        
        rbboxes = np.vstack(rbbox_result)

        # draw rbbox
        labels = [
            np.full(rbbox.shape[0], i, dtype=np.int32)
                for i, rbbox in enumerate(rbbox_result)
        ]
        labels = np.concatenate(labels)
        
        img = imread(img_show)
        
        scores = rbboxes[:, -1]
        inds = scores > score_thr
        rbboxes = rbboxes[inds, :8]
        labels = labels[inds]
        
        rbbox_color = (0, 255, 0)
        text_color = (0, 255, 0)
        font_scale = 0.5
        
        for rbbox, score, label in zip(rbboxes, scores, labels):
            rbbox_int = rbbox.astype(np.int32)
            rbbox_int = rbbox_int.reshape(4,2)
            cv2.drawContours(img,[rbbox_int],0,rbbox_color,1)
            label_text = class_names[
                label] if class_names is not None else 'cls {}'.format(label)
            label_text += '|{:.02f}'.format(score)
            cv2.putText(img, label_text, (rbbox_int[0][0], rbbox_int[0][1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        
        cv2.imwrite(file_name,img)

import colorsys
import random
 
def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors
 
def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
 
    return rgb_colors

def show_rbbox_color(data,result, img_norm_cfg, class_names,
            score_thr=0.1, file_name='0.png'):
    
    rbbox_result = result
    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        
        rbboxes = np.vstack(rbbox_result)

        # draw rbbox
        labels = [
            np.full(rbbox.shape[0], i, dtype=np.int32)
                for i, rbbox in enumerate(rbbox_result)
        ]
        labels = np.concatenate(labels)
        
        img = imread(img_show)
        
        scores = rbboxes[:, -1]
        inds = scores > score_thr
        rbboxes = rbboxes[inds, :8]
        labels = labels[inds]
        
        # rbbox_color = ncolors(16)
        rbbox_color = [[247, 11, 11], [244, 103, 19], \
            [250, 199, 48], [220, 245, 45], [150, 247, 52], \
                [69, 244, 44], [18, 243, 75], [27, 251, 167], \
                    [38, 248, 248], [18, 158, 242], [15, 74, 249], \
                        [33, 2, 253], [147, 44, 250], [220, 29, 248], \
                            [243, 16, 186], [250, 43, 121]]
        text_color = (0, 255, 0)
        font_scale = 0.5
        
        for rbbox, score, label in zip(rbboxes, scores, labels):
            rbbox_int = rbbox.astype(np.int32)
            rbbox_int = rbbox_int.reshape(4,2)
            cv2.drawContours(img,[rbbox_int],0,rbbox_color[label],2)
            # label_text = class_names[
            #     label] if class_names is not None else 'cls {}'.format(label)
            # label_text += '|{:.02f}'.format(score)
            # cv2.putText(img, label_text, (rbbox_int[0][0], rbbox_int[0][1] - 2),
            #         cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        
        cv2.imwrite(file_name,img)

def show_mask(data,result, img_norm_cfg, class_names,
            score_thr=0.3, file_name='0.png'):
    
    bbox_result, segm_result = result
    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        
        bboxes  = np.vstack(bbox_result)

        # draw segmentation masks
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            for i in inds:
                color_mask = np.random.randint(
                    0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
        # draw bbox
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        
        img = imread(img_show)
        
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        
        bbox_color = (0, 255, 0)
        text_color = (0, 255, 0)
        font_scale = 0.5
        
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            cv2.rectangle(img, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), bbox_color, 1)
            label_text = class_names[
                label] if class_names is not None else 'cls {}'.format(label)
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        
        cv2.imwrite(file_name,img)

def trans2ms_result(result):
    result = list(result)
    for i in range(len(result[0])): # 遍历每一个类别
        for j in range(len(result[0][i])): # 遍历每一个框
            result[0][i][j][-1] = result[1][1][i][j] 
            result[2][i][j][-1] = result[3][1][i][j]
    result[1] = result[1][0]
    result[3] = result[3][0]
    result = tuple(result)
    return result

def trans2mask_score(results):
    ms_results = []
    for result in results:
        result = list(result)
        for i in range(len(result[0])):
            for j in range(len(result[0][i])):
                result[0][i][j][-1] = result[1][1][i][j] 
        result[1] = result[1][0]
        result = tuple(result)
        ms_results.append(result)
    return ms_results

def trans2ms_results(results):
    ms_results = []
    for result in results:
        result = list(result)
        for i in range(len(result[0])):
            for j in range(len(result[0][i])):
                result[0][i][j][-1] = result[1][1][i][j] 
                result[2][i][j][-1] = result[3][1][i][j]
        result[1] = result[1][0]
        result[3] = result[3][0]
        result = tuple(result)
        ms_results.append(result)
    return ms_results

def trans2mask_results_V2(results):
    mask_results = []
    for result in results:
        result = list(result)
        result[1] = result[1][0]
        result = tuple(result)
        mask_results.append(result)
    return mask_results

def trans2mask_results(results):
    mask_results = []
    for result in results:
        result = list(result)
        result[1] = result[1][0]
        result[3] = result[3][0]
        result = tuple(result)
        mask_results.append(result)
    return mask_results

def trans2obb_results(results):
    obb_results = []
    for result in results:
        result = list(result)
        result = [result[0], result[-1]]
        result = tuple(result)
        obb_results.append(result)
    return obb_results

def trans2hbb_results(results):
    hbb_results = []
    for result in results:
        result = list(result)
        result = [result[0], result[1]]
        result = tuple(result)
        hbb_results.append(result)
    return hbb_results

def trans2mix_results(results):
    mix_results = []
    for result in results:
        result = list(result)
        for i in range(len(result[0])):
            for j in range(len(result[0][i])):
                ms_hbb = result[0][i][j][-1]
                ms_obb = result[2][i][j][-1]
                if ms_hbb < ms_obb:
                    result[0][i][j][-1] = ms_obb
                    result[1][i][j] = result[3][i][j]
        result = result[:2]
        result = tuple(result)
        mix_results.append(result)
    return mix_results

def show_bbox(data,result, img_norm_cfg, class_names,
            score_thr=0.3, file_name='0.png'):
    
    bbox_result = result
    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        
        bboxes  = np.vstack(bbox_result)


        # draw rbbox
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        
        img = imread(img_show)
        
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        
        bbox_color = (0, 255, 0)
        text_color = (0, 255, 0)
        font_scale = 0.5
        
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            cv2.rectangle(img, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), bbox_color, 1)
            label_text = class_names[
                label] if class_names is not None else 'cls {}'.format(label)
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        
        cv2.imwrite(file_name,img)