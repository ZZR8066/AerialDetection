import torch
import torch.nn as nn
from ..utils import ConvModule
from ..registry import NECKS
from ..utils import ConvModule
from mmdet.core import multi_apply

'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

@NECKS.register_module
class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_levels, with_r=False, 
                    conv_cfg=None, norm_cfg=None, activation=None):
                    
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        
        self.convs = nn.ModuleList()
        for _ in range(num_levels):
            conv = ConvModule(
                        in_size,
                        out_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=activation,
                        inplace=False)
            self.convs.append(conv)
            
    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.convs)
        
    def forward_single(self, x, conv):
        ret = self.addcoords(x)
        ret = conv(ret)
        return ret, None