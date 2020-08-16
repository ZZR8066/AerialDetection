import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..registry import NECKS
from ..utils import ConvModule
from .fpn_carafe import FPN_CARAFE
from mmdet.ops.carafe import CARAFEPack

@NECKS.register_module
class PAFPN_CARAFE(FPN_CARAFE):

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
        super(PAFPN_CARAFE, self).__init__(in_channels, out_channels, 
                                num_outs, start_level,
                                end_level, relu_before_extra_convs, 
                                conv_cfg, norm_cfg, activation, upsample_cfg)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(num_outs-1):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            
            self.downsample_convs.append(d_conv)
            
        for i in range(num_outs):
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.pafpn_convs.append(pafpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

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
            
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(len(self.fpn_convs))
        ]
        
        # part 2: add bottom-up path
        for i in range(len(self.downsample_convs)):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])
            
        # build outputs
        outs = [
            self.pafpn_convs[i](inter_outs[i]) for i in range(len(self.pafpn_convs))
        ]
        
        return tuple(outs)