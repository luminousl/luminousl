import os
import onnx
import torch
import numpy as np
import onnx_graphsurgeon as gs
import onnxruntime as ort
from onnxruntime_extensions import PyOp, onnx_op, PyOrtFunction
from bev_pool_v2 import bev_pool_v2 as bev_pool_v2_ext

import MultiScaleDeformableAttention as MSDA

# onnxruntime_extensions._extensions_pydll.PyCustomOpDef()
# @onnx_op(op_type="GPT2Tokenizer",
#             inputs=[PyCustomOpDef.dt_string],
#             outputs=[PyCustomOpDef.dt_int64, PyCustomOpDef.dt_int64],
#             attrs={"padding_length": PyCustomOpDef.dt_int64})
# def bpe_tokenizer(s, **kwargs):
#     padding_length = kwargs["padding_length"]
#     input_ids, attention_mask = cls.tokenizer.tokenizer_sentence([s[0]], padding_length)
#     return input_ids, attention_mask

### ort自定义算子 - 矩阵求逆 示例
@onnx_op(op_type="Inverse", inputs=[PyOp.dt_float])
def Inverse(x):
    # the user custom op implementation here:
    return np.linalg.inv(x)

### BEV_OD任务中FastMSDA的ort算子实现
@onnx_op(op_type="FastMSDA_BEVOD", 
            inputs=[PyOp.dt_float, PyOp.dt_float, PyOp.dt_float, PyOp.dt_float, PyOp.dt_float],
            outputs=[PyOp.dt_float])
def FastMSDA_BEVOD(sampling_locations, attention_weights, camera, radar, lidar):
    # the user custom op implementation here:
    im2col_step = 64
    camera_cuda = torch.tensor(camera).cuda().permute(0,2,3,1).reshape([1,-1,8,32])
    radar_cuda = torch.tensor(radar).cuda().permute(0,2,3,1).reshape([1,-1,8,32])
    lidar_cuda = torch.tensor(lidar).cuda().permute(0,2,3,1).reshape([1,-1,8,32])
    value_cuda = torch.cat([camera_cuda, radar_cuda, lidar_cuda], axis=1)
    # breakpoint()
    value_spatial_shapes_cuda = torch.tensor([[64,120],[64,120],[64,120],[64,120],[64,120],[64,120],[64,120],[190,70],[190,70]], dtype=torch.int64).cuda()
    value_level_start_index_cuda = torch.tensor([0,7680,15360,23040,30720,38400,46080,53760,67060], dtype=torch.int64).cuda()
    sampling_locations_cuda = torch.tensor(sampling_locations).cuda()
    attention_weights_cuda = torch.tensor(attention_weights).cuda()
    output = MSDA.ms_deform_attn_forward(value_cuda, 
                                        value_spatial_shapes_cuda, 
                                        value_level_start_index_cuda, 
                                        sampling_locations_cuda, 
                                        attention_weights_cuda, 
                                        im2col_step)
    return output.cpu().numpy()


### v9 BEV_OD任务中FastMSDA的ort算子实现
@onnx_op(op_type="FastMSDA_BEVOD_v9", 
            inputs=[PyOp.dt_float, PyOp.dt_float, PyOp.dt_float, PyOp.dt_float, PyOp.dt_float, PyOp.dt_float],
            outputs=[PyOp.dt_float])
def FastMSDA_BEVOD_v9(sampling_locations, attention_weights, camera_7v, camera_4v, radar, lidar):
    # the user custom op implementation here:
    im2col_step = 64
    camera_7v_cuda = torch.tensor(camera_7v).cuda().permute(0,2,3,1).reshape([1,-1,8,32])
    camera_4v_cuda = torch.tensor(camera_4v).cuda().permute(0,2,3,1).reshape([1,-1,8,32])
    radar_cuda = torch.tensor(radar).cuda().permute(0,2,3,1).reshape([1,-1,8,32])
    lidar_cuda = torch.tensor(lidar).cuda().permute(0,2,3,1).reshape([1,-1,8,32])
    value_cuda = torch.cat([camera_7v_cuda, camera_4v_cuda, radar_cuda, lidar_cuda], axis=1)
    value_spatial_shapes_cuda = torch.tensor([[64,120],[64,120],[64,120],[64,120],[64,120],[64,120],[64,120],[64,80],[64,80],[64,80],[64,80],[200,70],[200,70]], dtype=torch.int64).cuda()
    value_level_start_index_cuda = torch.tensor([0,7680,15360,23040,30720,38400,46080,53760,58880,64000,69120,74240,88240], dtype=torch.int64).cuda()
    sampling_locations_cuda = torch.tensor(sampling_locations).cuda()
    attention_weights_cuda = torch.tensor(attention_weights).cuda()
    output = MSDA.ms_deform_attn_forward(value_cuda, 
                                        value_spatial_shapes_cuda, 
                                        value_level_start_index_cuda, 
                                        sampling_locations_cuda, 
                                        attention_weights_cuda, 
                                        im2col_step)
    return output.cpu().numpy()

@onnx_op(op_type="FastMSDA_STATIC", 
            inputs=[PyOp.dt_float, PyOp.dt_float, PyOp.dt_float],
            outputs=[PyOp.dt_float],
            attrs={
                        "feature_layout": PyOp.dt_string,
                        "spatial_shape": PyOp.dt_string,
                        "static_msda": PyOp.dt_int64,
                    }
            )
def FastMSDA_STATIC(sampling_locations, attention_weights, camera, **kwargs):
    # the user custom op implementation here:
    static_msda = kwargs.get("static_msda")
    assert static_msda == 1
    im2col_step = 64
    # camera_cuda = torch.tensor(camera).cuda().permute(0,3,2,1).reshape([1,-1,int(camera.shape[1]/32),32])
    camera_cuda = torch.tensor(camera).cuda().permute(0,3,2,1).reshape([1,-1,torch.tensor(sampling_locations).shape[2], int(camera.shape[1]/torch.tensor(sampling_locations).shape[2])])
    spatial_shape_str = kwargs.get("spatial_shape")
    spatial_shape = list(map(int, spatial_shape_str.split('_')))
    # assert spatial_shape[0] <= spatial_shape[1]
    value_spatial_shapes_cuda = torch.tensor([[spatial_shape[0],spatial_shape[1]]], dtype=torch.int64).cuda() #[
    value_level_start_index_cuda = torch.tensor([0], dtype=torch.int64).cuda()
    sampling_locations_cuda = torch.tensor(sampling_locations).cuda()
    attention_weights_cuda = torch.tensor(attention_weights).cuda()
    output = MSDA.ms_deform_attn_forward(camera_cuda, 
                                        value_spatial_shapes_cuda, 
                                        value_level_start_index_cuda, 
                                        sampling_locations_cuda, 
                                        attention_weights_cuda, 
                                        im2col_step)
    print(f"FastMSDA_STATIC output shape {output.shape}")
    return output.cpu().numpy()


### TRAFFICLANE_BEV任务中transformer2 FastMSDA的ort算子实现
@onnx_op(op_type="FastMSDA_transformer2", 
            inputs=[PyOp.dt_float, PyOp.dt_float, PyOp.dt_float, PyOp.dt_float],
            outputs=[PyOp.dt_float])
def FastMSDA_transformer2(sampling_offsets, attention_weights, value, reference_points):
    # the user custom op implementation here:
    im2col_step = 64
    value_cuda = torch.tensor(value).cuda()
    value_cuda = value_cuda.permute(0, 2, 3, 1).reshape(1, -1, value.shape[1])
    # swapped_value_cuda = value_cuda[[4, 5, 6, 0, 1, 2, 3]]
    # value_cuda = swapped_value_cuda.flatten(2).transpose(1, 2).reshape(1, -1, swapped_value_cuda.size(1))
    value_cuda = value_cuda.repeat(1, 4, 1)
    value_cuda = value_cuda.view(1, -1, 8, 32)
    n_heads = 8
    n_levels = 28
    n_points = 8
    sampling_offsets = sampling_offsets.reshape(sampling_offsets.shape[0], sampling_offsets.shape[1], n_heads, n_levels, n_points, 2)
    # new_shape = (sampling_offsets.shape[0], sampling_offsets.shape[1], 
    #         n_heads, n_levels, n_points, 2)
    # sampling_offsets = sampling_offsets.view(new_shape)
    sampling_offsets_cuda = torch.tensor(sampling_offsets).cuda()


    value_spatial_shapes_cuda = torch.tensor([[64,120],[64,120],[64,120],[64,120],[64,120],[64,120],[64,120],
        [64,120],[64,120],[64,120],[64,120],[64,120],[64,120],[64,120],
        [64,120],[64,120],[64,120],[64,120],[64,120],[64,120],[64,120],
        [64,120],[64,120],[64,120],[64,120],[64,120],[64,120],[64,120]], dtype=torch.int64).cuda()
    # value_level_start_index_cuda = torch.tensor([0,   7680,  15360,  23040,  30720,  38400,  46080,  53760,  61440,
    #     69120,  76800,  84480,  92160,  99840, 107520, 115200, 122880, 130560,
    #     138240, 145920, 153600, 161280, 168960, 176640, 184320, 192000, 199680,
    #     207360], dtype=torch.int64).cuda()
    value_level_start_index_cuda = torch.tensor([0,  7680, 15360, 23040, 30720, 38400, 46080,     0,  7680, 15360,
        23040, 30720, 38400, 46080,     0,  7680, 15360, 23040, 30720, 38400,
        46080,     0,  7680, 15360, 23040, 30720, 38400, 46080], dtype=torch.int64).cuda()


    attention_weights_cuda = torch.tensor(attention_weights).cuda()

    reference_points_cuda = torch.tensor(reference_points).cuda()

    reference_points_cuda[..., 0] += 0.5  # X维度
    reference_points_cuda[..., 1] += 0.5 # Y维度
    reference_points_cuda[..., 0] /= 120  # X维度
    reference_points_cuda[..., 1] /= 64 # Y维度

    # N, Len_q, n_heads, n_levels, n_points, 2
    if reference_points_cuda.shape[-1] == 2:
        offset_normalizer = torch.stack([value_spatial_shapes_cuda[..., 1], value_spatial_shapes_cuda[..., 0]], -1)
        sampling_locations = reference_points_cuda[:, :, None, :, None, :] \
                            + sampling_offsets_cuda / offset_normalizer[None, None, None, :, None, :]
    elif reference_points_cuda.shape[-1] == 4:
        sampling_locations = reference_points_cuda[:, :, None, :, None, :2] \
                            + sampling_offsets_cuda / n_points * reference_points_cuda[:, :, None, :, None, 2:] * 0.5
    else:
        raise ValueError(
            'Last dim of reference_points_cuda must be 2 or 4, but get {} instead.'.format(reference_points_cuda.shape[-1]))
    # breakpoint()
    sampling_locations_cuda = torch.tensor(sampling_locations).cuda()
    output = MSDA.ms_deform_attn_forward(value_cuda, 
                                        value_spatial_shapes_cuda, 
                                        value_level_start_index_cuda, 
                                        sampling_locations_cuda, 
                                        attention_weights_cuda, 
                                        im2col_step)
    # breakpoint()
    return output.cpu().numpy()


### 原版MultiscaleDeformableAttnPlugin_TRT的ort算子实现
@onnx_op(op_type="MultiscaleDeformableAttnPlugin_TRT", 
            inputs=[PyOp.dt_float, PyOp.dt_int32, PyOp.dt_int32, PyOp.dt_float, PyOp.dt_float],
            outputs=[PyOp.dt_float])
def MultiscaleDeformableAttnPlugin_TRT(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
    # the user custom op implementation here:
    im2col_step = 64
    value_cuda = torch.tensor(value).cuda()
    # 关键修复：确保int64类型
    if value_spatial_shapes.dtype != np.int64:
        value_spatial_shapes = value_spatial_shapes.astype(np.int64)
    value_spatial_shapes_cuda = torch.tensor(value_spatial_shapes).cuda()
    
    if value_level_start_index.dtype != np.int64:
        value_level_start_index = value_level_start_index.astype(np.int64)
    value_level_start_index_cuda = torch.tensor(value_level_start_index).cuda()

    # value_spatial_shapes_cuda = torch.tensor(value_spatial_shapes).cuda()
    # value_level_start_index_cuda = torch.tensor(value_level_start_index).cuda()
    sampling_locations_cuda = torch.tensor(sampling_locations).cuda()
    attention_weights_cuda = torch.tensor(attention_weights).cuda()
    output = MSDA.ms_deform_attn_forward(value_cuda, value_spatial_shapes_cuda, 
                                        value_level_start_index_cuda, sampling_locations_cuda, 
                                        attention_weights_cuda, im2col_step)
    return output.cpu().numpy()


### bev_pool_v2的ort算子实现
@onnx_op(op_type="bev_pool_v2", 
            inputs=[PyOp.dt_float, PyOp.dt_float, PyOp.dt_int32, PyOp.dt_int32, PyOp.dt_int32, PyOp.dt_int32, PyOp.dt_int32],
            outputs=[PyOp.dt_float],
            attrs={"out_height": PyOp.dt_int64, "out_width": PyOp.dt_int64, })
def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                interval_starts, interval_lengths, **kwargs):
    # the user custom op implementation here:
    depth_cuda = torch.tensor(depth).unsqueeze(0).cuda()
    feat_cuda = torch.tensor(feat).unsqueeze(0).cuda()
    ranks_depth_cuda = torch.tensor(ranks_depth).cuda()
    ranks_feat_cuda = torch.tensor(ranks_feat).cuda()
    ranks_bev_cuda = torch.tensor(ranks_bev).cuda()
    interval_starts_cuda = torch.tensor(interval_starts).cuda()
    interval_lengths_cuda = torch.tensor(interval_lengths).cuda()
    out_height = kwargs["out_height"]
    out_width = kwargs["out_width"]
    out_channel = feat.shape[-1]
    bev_feat_shape = (1, 1, out_height, out_width, out_channel)

    out = bev_pool_v2_ext(depth_cuda, feat_cuda, ranks_depth_cuda, ranks_feat_cuda, ranks_bev_cuda,
                               bev_feat_shape, interval_starts_cuda,
                               interval_lengths_cuda)
    return out.squeeze(0).cpu().numpy()