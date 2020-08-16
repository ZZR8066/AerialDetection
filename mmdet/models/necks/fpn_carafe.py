import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, build_upsample_layer

from mmdet.ops.carafe import CARAFEPack

from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class FPN_CARAFE(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 upsample_cfg=dict(
                     type='carafe',
                     up_kernel=5,
                     up_group=1,
                     encoder_kernel=3,
                     encoder_dilation=1)):
        super(FPN_CARAFE, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        
        self.upsample_cfg = upsample_cfg.copy()
        self.upsample = self.upsample_cfg.get('type')
        
        assert self.upsample in [
            'nearest', 'bilinear', 'deconv', 'pixel_shuffle', 'carafe', None
        ]
        if self.upsample in ['deconv', 'pixel_shuffle']:
            assert hasattr(
                self.upsample_cfg,
                'upsample_kernel') and self.upsample_cfg.upsample_kernel > 0
            self.upsample_kernel = self.upsample_cfg.pop('upsample_kernel')
            
        
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.upsample_modules = nn.ModuleList()
        
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            if i != self.backbone_end_level - 1:
                upsample_cfg_ = self.upsample_cfg.copy()
                if self.upsample == 'deconv':
                    upsample_cfg_.update(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=self.upsample_kernel,
                        stride=2,
                        padding=(self.upsample_kernel - 1) // 2,
                        output_padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'pixel_shuffle':
                    upsample_cfg_.update(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        scale_factor=2,
                        upsample_kernel=self.upsample_kernel)
                elif self.upsample == 'carafe':
                    upsample_cfg_.update(channels=out_channels, scale_factor=2)
                else:
                    # suppress warnings
                    align_corners = (None
                                     if self.upsample == 'nearest' else False)
                    upsample_cfg_.update(
                        scale_factor=2,
                        mode=self.upsample,
                        align_corners=align_corners)
                upsample_module = build_upsample_layer(upsample_cfg_)
                self.upsample_modules.append(upsample_module)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_out_levels = (
            num_outs - self.backbone_end_level + self.start_level)
        if extra_out_levels >= 1:
            for i in range(extra_out_levels):
                in_channels = (
                    self.in_channels[self.backbone_end_level -
                                     1] if i == 0 else out_channels)
                extra_l_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                if self.upsample == 'deconv':
                    upsampler_cfg_ = dict(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=self.upsample_kernel,
                        stride=2,
                        padding=(self.upsample_kernel - 1) // 2,
                        output_padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'pixel_shuffle':
                    upsampler_cfg_ = dict(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        scale_factor=2,
                        upsample_kernel=self.upsample_kernel)
                elif self.upsample == 'carafe':
                    upsampler_cfg_ = dict(
                        channels=out_channels,
                        scale_factor=2,
                        **self.upsample_cfg)
                else:
                    # suppress warnings
                    align_corners = (None
                                     if self.upsample == 'nearest' else False)
                    upsampler_cfg_ = dict(
                        scale_factor=2,
                        mode=self.upsample,
                        align_corners=align_corners)
                upsampler_cfg_['type'] = self.upsample
                upsample_module = build_upsample_layer(upsampler_cfg_)
                extra_fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.upsample_modules.append(upsample_module)
                self.fpn_convs.append(extra_fpn_conv)
                self.lateral_convs.append(extra_l_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def slice_as(self, src, dst):
        """Slice ``src`` as ``dst``

        Note:
            ``src`` should have the same or larger size than ``dst``.

        Args:
            src (torch.Tensor): Tensors to be sliced.
            dst (torch.Tensor): ``src`` will be sliced to have the same
                size as ``dst``.

        Returns:
            torch.Tensor: Sliced tensor.
        """
        assert (src.size(2) >= dst.size(2)) and (src.size(3) >= dst.size(3))
        if src.size(2) == dst.size(2) and src.size(3) == dst.size(3):
            return src
        else:
            return src[:, :, :dst.size(2), :dst.size(3)]
            
    def tensor_add(self, a, b):
        """Add tensors ``a`` and ``b`` that might have different sizes"""
        if a.size() == b.size():
            c = a + b
        else:
            c = a + self.slice_as(b, a)
        return c
        
    def forward(self, inputs):
        """Forward function"""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i <= self.backbone_end_level - self.start_level:
                input = inputs[min(i + self.start_level, len(inputs) - 1)]
            else:
                input = laterals[-1]
            lateral = lateral_conv(input)
            laterals.append(lateral)

        # build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            if self.upsample is not None:
                upsample_feat = self.upsample_modules[i - 1](laterals[i])
            else:
                upsample_feat = laterals[i]
            laterals[i - 1] = self.tensor_add(laterals[i - 1], upsample_feat)

        # build outputs
        num_conv_outs = len(self.fpn_convs)
        outs = []
        for i in range(num_conv_outs):
            out = self.fpn_convs[i](laterals[i])
            outs.append(out)
        return tuple(outs)
