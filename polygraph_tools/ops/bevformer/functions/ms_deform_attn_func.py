import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import logging

try:
    import MultiScaleDeformableAttention as MSDA
except ImportError as e:
    logging.warning('MultiScaleDeformableAttention package is not installed correctly due to {}. '
                    'Make sure you do not need to run MSDA.'.format(e))


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
                im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
            ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
                grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class MSDeformAttnFunctionTrt(Function):
    @staticmethod
    def symbolic(
        g,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_offsets,
        attention_weights,
        reference_points,
        valid_index,
    ):
        return g.op(
            "custom::MSDAPlugin",
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_offsets,
            attention_weights,
            reference_points,
            valid_index,
            channels_i = 32,
            max_eff_camera_level_i = 42,
            spatial_height_i = 64,
            spatial_width_i = 120,
            use_camera_num_i = 7,
            channel_first_i = 1,
            layout_s = "HWC8",
        )

    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_offsets,
        attention_weights,
        reference_points,
        valid_index,
    ):
        # N, Len_q, n_heads, n_levels, n_points, 2
        # if torch.onnx.is_in_onnx_export():
        #     value = value.view(value.shape[0], -1, value.shape[4], value.shape[5])
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([value_spatial_shapes[..., 1], value_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        ctx.im2col_step = 64
        ctx.fp16 = False
        if value.dtype == torch.float16:
            ctx.fp16 = True
            value = value.float()
            sampling_locations = sampling_locations.float()
            attention_weights = attention_weights.float()
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
            ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)

        return output.half() if ctx.fp16 else output


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)

    stacked = torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
    output = stacked.sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


def ms_deform_attn_core_pytorch_onnx(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        print('{} sampling_grid_l_'.format(lid_), sampling_grid_l_.size())
        # # N_*M_, D_, Lq_, P_
        # sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
        #                                   mode='bilinear', padding_mode='zeros', align_corners=False)
        value_l_ = value_l_[:, :, :4, :2]
        sampling_value_l_ = torch.cat((value_l_, sampling_grid_l_), dim=1)
        sampling_value_l_ = sampling_value_l_.transpose(1, 2).repeat(1, 8, 1, 2)[:, :, :1600, :]
        sampling_value_list.append(sampling_value_l_)

    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)

    stacked = torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
    output = stacked.sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()
