from __future__ import division

import torch
import torch.nn as nn

from mmdet import ops
from ..registry import ROI_EXTRACTORS
import pdb

@ROI_EXTRACTORS.register_module
class ARboxMultiRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 w_enlarge=1.2,
                 h_enlarge=1.4,
                 ratio_max=5.0):
        super(ARboxMultiRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.w_enlarge = w_enlarge
        self.h_enlarge = h_enlarge
        self.ratio_max = ratio_max

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def get_poolwh(self, rois, base_size):
        ratios = rois[:, 3] / rois[:, 4]
        assert ratios.min() >= 1.0
        ratios = ratios.ceil()
        ratio  = ratios.max()
        ratio  = min(ratio, self.ratio_max)
        pool_h = int(base_size)
        pool_w = int(ratio * base_size)
        return pool_w, pool_h
    
    def forward(self, feats, rois):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        base_size = out_size
        out_w, out_h = self.get_poolwh(rois, base_size)
        
        num_levels = len(feats)
        roi_feats=[]
        for i in range(num_levels):
            roi_feats_t = self.roi_layers[i](feats[i], rois, out_w, out_h)
            roi_feats.append(roi_feats_t)
        
        # max pool
        feature_size = roi_feats[0].size()
        roi_feats = [var.view(var.size(0),-1) for var in roi_feats]
        for i in range(1, num_levels):
            roi_feats[0] = torch.max(roi_feats[0], roi_feats[i])
        roi_feats = roi_feats[0]
        roi_feats = roi_feats.view(feature_size)
        
        return roi_feats