import warnings
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from zpilot_nn.ops.bevformer.functions import MSDeformAttnFunction, MSDeformAttnFunctionTrt
from zpilot_nn.models.attention.multi_scale_deformable_attn_function import MSDeformAttnFunctionV2Trt


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, global_config, d_model=256, n_levels=4, n_heads=8, n_points=4, with_bias=True, use_share_input_proj=False, valid_weight_last=True):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.with_bias = with_bias

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.use_share_input_proj = use_share_input_proj
        self.valid_weight_last = valid_weight_last
        if not self.use_share_input_proj:
            self.value_proj = nn.Linear(d_model, d_model, bias=with_bias)
            self.value_proj_1x1conv = nn.Conv2d(d_model, d_model, kernel_size = 1,bias=with_bias)
        self.output_proj = nn.Linear(d_model, d_model, bias=with_bias)
        self.global_config = global_config

        self._reset_parameters()

    def reset_weight(self):
        # TODO: support in reset_weight.py
        if not self.use_share_input_proj:
            with torch.no_grad():
                self.value_proj_1x1conv.weight.copy_(self.value_proj.weight.view(self.d_model, self.d_model, 1, 1)) # Reshape MLP weight to Conv1x1 weight shape
                if self.with_bias:
                    self.value_proj_1x1conv.bias.copy_(self.value_proj.bias)

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        if not self.use_share_input_proj:
            xavier_uniform_(self.value_proj.weight.data)
        # constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        # constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                valid_index, input_padding_mask=None, valid_weight=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        if not torch.onnx.is_in_onnx_export() and not self.global_config.dump_calibset:
            N, Len_in, _ = input_flatten.shape
        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        if not self.use_share_input_proj:
            if torch.onnx.is_in_onnx_export() or self.global_config.dump_calibset:
                value = self.value_proj_1x1conv(input_flatten)
            else:
                value = self.value_proj(input_flatten)
        else:
            value = input_flatten
        # value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        # if not torch.onnx.is_in_onnx_export():
        if not torch.onnx.is_in_onnx_export() and not self.global_config.dump_calibset:
            value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # else:
        #     value = value.view(N, self.cam_num, self.feat_h, self.feat_w, self.n_heads, self.d_model // self.n_heads)
        if not torch.onnx.is_in_onnx_export():
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        else:
            sampling_offsets = self.sampling_offsets(query)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # # N, Len_q, n_heads, n_levels, n_points, 2
        # if reference_points.shape[-1] == 2:
        #     offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        #     sampling_locations = reference_points[:, :, None, :, None, :] \
        #                          + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        # elif reference_points.shape[-1] == 4:
        #     sampling_locations = reference_points[:, :, None, :, None, :2] \
        #                          + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        # else:
        #     raise ValueError(
        #         'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        if not torch.onnx.is_in_onnx_export() and not self.global_config.dump_calibset:
            # N, Len_q, n_heads, n_levels, n_points, 2
            if reference_points.shape[-1] == 2:
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points[:, :, None, :, None, :] \
                                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            elif reference_points.shape[-1] == 4:
                sampling_locations = reference_points[:, :, None, :, None, :2] \
                                    + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            else:
                raise ValueError(
                    'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
                self.im2col_step)
        else:
            #valud [bs, feat_len, num_heads, dim/num_head]
            #spatial_shapes [l, 2]
            #level_start_index [l, ]
            #sampling_locations [bs, num_query, num_heads, l, num_point, 2]
            #attention_weights [bs, num_query, num_heads, l, num_point]
            # spatial_height = input_spatial_shapes[0][0]
            # spatial_width = input_spatial_shapes[0][1]
            # num_cameras = Len_in // (spatial_height * spatial_width)
            # value = value.view(N, num_cameras, spatial_height, spatial_width, self.n_heads, self.d_model // self.n_heads)
            output = MSDeformAttnFunctionV2Trt.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_offsets, attention_weights, reference_points,
                self.n_heads, self.d_model,self.n_points,[value.shape[2],value.shape[3]]*reference_points.shape[2],reference_points.shape).flatten(2)
        if not self.valid_weight_last:
            output = output / valid_weight
        output = self.output_proj(output)
        return output
