import torch.nn as nn

from .rbbox_head import BBoxHeadRbbox
from ..registry import HEADS
from ..utils import ConvModule

from .convfc_rbbox_head import SharedFCBBoxHeadRbbox

@HEADS.register_module
class FusedFCBBoxHeadRbbox(SharedFCBBoxHeadRbbox):

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                # last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
                if isinstance(self.roi_feat_size, int):
                    last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
                elif isinstance(self.roi_feat_size, tuple):
                    assert len(self.roi_feat_size) == 2
                    assert isinstance(self.roi_feat_size[0], int)
                    assert isinstance(self.roi_feat_size[1], int)
                    last_layer_dim *= (self.roi_feat_size[0] * self.roi_feat_size[1])
            for i in range(num_branch_fcs):
                fc_in_channels = self.fc_out_channels
                # fc_in_channels = (
                #     last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim