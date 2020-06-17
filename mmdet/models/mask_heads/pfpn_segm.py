import cv2
import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from torch import sqrt

@HEADS.register_module
class PFPNSegmHead(nn.Module):

    def __init__(self,
                num_classes=15):

        super(PFPNSegmHead, self).__init__()
        self.num_classes = num_classes
		# Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128)
        self.gn2 = nn.GroupNorm(256, 256)
    
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(torch.tensor(2.) / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
     
    def forward(self, x):
        p2, p3, p4, p5, _ = x
        # Semantic
        _, _, h, w = p2.size()
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), int(h/4), int(w/4))
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), int(h/2), int(w/2))
        # 256->128
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), int(h/2), int(w/2))
        # 256->128
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)
        
        s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        return self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w)
        
    def get_target(self, img_shape, gt_labels_list, gt_masks_list):
        mask_targets_list = []
        for gt_labels, gt_masks in zip(gt_labels_list, gt_masks_list):
            mask_targets = []
            for label in range(1,self.num_classes+1):
                flag = (gt_labels == label).cpu().numpy()
                mask_label = np.zeros(img_shape,dtype='uint8')
                if flag.any():
                    masks = gt_masks[flag]
                    for mask in masks:
                        mask = np.pad(mask,((0,img_shape[0]-mask.shape[0]), (0,img_shape[1]-mask.shape[1])),
                                'constant',constant_values = (0,0))
                        mask = mask.astype(np.bool)
                        mask_label[mask] = 1
                mask_targets.append(mask_label)
            mask_targets_list.append(torch.from_numpy(
                np.stack(mask_targets)).float().to(gt_labels.device))
        return torch.cat(list(mask_targets_list))
        
    # Sigmoid Fcoal Loss
    def loss( self,
              pred,
              target,
              weight=1.0,
              gamma=2.0,
              alpha=0.25,
              reduction='mean'):
        loss = dict()
        pred_sigmoid = pred.sigmoid()
        target = target.view_as(pred)
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
        weight = weight * pt.pow(gamma)
        loss_mask = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * weight
        loss['loss_segm'] = loss_mask.mean()
        return loss