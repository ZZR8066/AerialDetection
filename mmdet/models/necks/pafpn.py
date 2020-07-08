import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..utils import ConvModule
from ..registry import NECKS

@NECKS.register_module
class PAFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None):
        super(PAFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None
        self.panet_buttomup=True

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
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                # normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                # normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.panet_buttomup_conv1_modules=nn.ModuleList()
        self.panet_buttomup_conv2_modules=nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level-1):
            l_conv_panet = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                # normalize=normalize,
                bias=self.with_bias,
                activation='relu',
                inplace=True)
            #pdb.set_trace()
            fpn_conv_panet = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                # normalize=normalize,
                bias=self.with_bias,
                activation='relu',
                inplace=True)

            self.panet_buttomup_conv1_modules.append(
                l_conv_panet
            )
            self.panet_buttomup_conv2_modules.append(
                fpn_conv_panet
            )


            # lvl_id = i - self.start_level
            # setattr(self, 'lateral_conv{}'.format(lvl_id), l_conv)
            # setattr(self, 'fpn_conv{}'.format(lvl_id), fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.backbone_end_level - 1]
                               if i == 0 else out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    # normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)




    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        out_panet=[]
        if self.panet_buttomup:
            out_panet.append(outs[0])
            for i in range(0, used_backbone_levels-1):
                fpn_tmp = self.panet_buttomup_conv1_modules[i](out_panet[-1])
                #print(fpn_middle_blobs[self.num_backbone_stages - i].size())
                #pdb.set_trace()
                fpn_tmp = fpn_tmp + outs[i+1]
                fpn_tmp = self.panet_buttomup_conv2_modules[i](fpn_tmp)
                out_panet.append(fpn_tmp)   
            outs=out_panet
            #pdb.set_trace()     
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
                    #pdb.set_trace()
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)