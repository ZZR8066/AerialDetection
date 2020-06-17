from torch.autograd import Function

from .. import roi_align_rotated_cuda

class ARoIAlignRotatedFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_w, out_h, base_size, spatial_scale, sample_num=0):
        out_h = int(out_h)
        out_w = int(out_w)
        
        ctx.out_h = out_h
        ctx.out_w = out_w
        ctx.base_size = base_size
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new_zeros(num_rois, num_channels, out_h, out_w)
        if features.is_cuda:
            roi_align_rotated_cuda.forward(features, rois, out_h, out_w, spatial_scale,
                                   sample_num, output)
        else:
            raise NotImplementedError

        return output

    @staticmethod
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        assert (feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = feature_size
        # out_w = grad_output.size(3)
        # out_h = grad_output.size(2)
        out_w = ctx.out_w
        out_h = ctx.out_h
        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height,
                                        data_width)
            roi_align_rotated_cuda.backward(grad_output.contiguous(), rois, out_h,
                                    out_w, spatial_scale, sample_num,
                                    grad_input)

        return grad_input, grad_rois, None, None, None, None, None


aroi_align_rotated = ARoIAlignRotatedFunction.apply