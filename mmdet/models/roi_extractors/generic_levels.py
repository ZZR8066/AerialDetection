from __future__ import division

from torch import nn

from mmdet import ops
from mmdet.models.plugins.non_local import NonLocal2D
from mmdet.models.plugins.generalized_attention import GeneralizedAttention
from mmdet.models.utils import ConvModule
from ..registry import ROI_EXTRACTORS

class NoProcess(object):
    """Apply identity function."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x


def SequentialModule(modules):
    """Apply a sequential module."""
    nn_modules = []
    for mc in modules:
        type_ = mc.pop('type')
        nn_modules.append(load_processing_model(type_, mc))
    return nn.Sequential(*nn_modules)


# Models supported by GRoIE
models = {
    'NonLocal2D': NonLocal2D,
    'GeneralizedAttention': GeneralizedAttention,
    'ConvModule': ConvModule,
    'Sequential': SequentialModule,
    'ReLU': nn.ReLU,
}


def load_processing_model(model_type, config):
    """Load a specific module."""
    return models.get(model_type, NoProcess)(**config)
    

@ROI_EXTRACTORS.register_module
class SumGenericRoiExtractor(nn.Module):
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
                 post_conf=None,
                 pre_conf=None):
        super(SumGenericRoiExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale

        pre_type = pre_conf.pop('type')
        post_type = post_conf.pop('type')

        # build pre/post processing modules
        self.post_conv = load_processing_model(post_type, post_conf)
        self.pre_conv = load_processing_model(pre_type, pre_conf)
        self.relu = nn.ReLU(inplace=False)

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

    def forward(self, feats, rois):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, out_size, out_size)
        for i in range(num_levels):
            # apply pre-processing to a RoI extracted from each layer
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            roi_feats_t = self.pre_conv(roi_feats_t)
            roi_feats_t = self.relu(roi_feats_t)
            # and sum them all
            roi_feats += roi_feats_t

        # apply post-processing before return the result
        roi_feats = self.post_conv(roi_feats)
        return roi_feats