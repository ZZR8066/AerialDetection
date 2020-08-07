import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import weighted_smoothl1
from mmdet.core.bbox.geometry import rbbox_overlaps_cy_warp
from ..registry import LOSSES


@LOSSES.register_module
class IoUSmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, loss_weight=1.0):
        super(IoUSmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target, pred_bboxes, target_bboxes, weight):
        regression_loss = smooth_l1_loss(pred, target, beta=self.beta, reduction='none')
        regression_loss = regression_loss.sum(dim=1).reshape(-1,1)

        regression_loss = regression_loss / regression_loss.data * weight

        # with torch.no_grad():
        assert pred_bboxes.shape[0] == target_bboxes.shape[0]
        overlaps = rbbox_overlaps_cy_warp(pred_bboxes.cpu().numpy(), target_bboxes)
        overlaps = overlaps.diag().reshape(-1,1).float()
        overlaps = overlaps.clamp(1e-4, 1.0)
        iou_factor = -1*torch.log(overlaps)
        # iou_factor = -1*torch.log(overlaps) / (regression_loss.data + 1e-4)

        # iou_factor = -1*torch.log(overlaps) / (regression_loss + 1e-4)
        # print(iou_factor.max(), '    ', iou_factor.min())
        # iou_factor = iou_factor.clamp(0.5,1.0)
        # normalizer = torch.max(torch.tensor(1.0).float(), torch.tensor(pred.shape[0]).float())

        regression_loss = regression_loss * iou_factor

        # loss_bbox = self.loss_weight * (torch.sum(regression_loss * iou_factor).mean()[None])

        loss_bbox = self.loss_weight * (regression_loss.mean()[None])

        return loss_bbox
    
# def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
#     if avg_factor is None:
#         avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
#     loss = smooth_l1_loss(pred, target, beta, reduction='none')
#     return torch.sum(loss * weight)[None] / avg_factor

def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()
