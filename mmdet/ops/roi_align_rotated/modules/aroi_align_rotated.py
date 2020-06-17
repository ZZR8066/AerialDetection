from torch.nn.modules.module import Module
from ..functions.aroi_align_rotated import ARoIAlignRotatedFunction


class ARoIAlignRotated(Module):

    def __init__(self, out_size, spatial_scale, sample_num=0):
        super(ARoIAlignRotated, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)

    def forward(self, features, rois, out_w, out_h):
        return ARoIAlignRotatedFunction.apply(features, rois, out_w, out_h, 
                    self.out_size, self.spatial_scale, self.sample_num)
