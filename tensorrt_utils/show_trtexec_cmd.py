import onnx
import onnx_graphsurgeon as gs
import argparse

###  
# 备注： 对于模型新加的输入输出，不用在下面的字典中添加新的字段了。会自动适配。
# 备注： 对于不是chw的输入输出，才需要特殊指定
###

thor_special_input_formats={
    'TS_od_feat_1':                     'fp16:hwc8',
    'TS_bev_static_1':                  'fp16:hwc8',
    'LK_lss_inp':                       'fp16:hwc8',
    'LK_trafficlane_2d_inp':            'fp16:hwc8',
    'LK_static_lidar_feat':             'fp16:hwc8',
    "LK_trafficlane_2d_feat":           'int8:chw32',
    "lidar_feat":                       'fp16:hwc8',
}

thor_special_output_formats={
    'TS_od_feat':                       'fp16:hwc8',
    'TS_bev_static':                    'fp16:hwc8',
    'LK_lss_inp':                       'fp16:hwc8',
    'LK_trafficlane_2d_inp':            'fp16:hwc8',
    'LK_static_lidar_feat':             'fp16:hwc8',
    "LK_trafficlane_2d_feat":           'int8:chw32',
}

orin_special_input_formats={
    "LK_trafficlane_2d_feat":           'int8:chw32',
}

orin_special_output_formats={
    "LK_trafficlane_2d_feat":           'int8:chw32',
}

def get_onnx_IOformat(graph):
    dtype_dict ={
        'float32': "fp32",
        'float16': "fp16",
        'int32': "int32",
        'int8': "int8",
        'bool': "bool",
    }
    Input_format = {}
    Output_format = {}
    for inp in graph.inputs:
        Input_format[inp.name] = f"{dtype_dict[str(inp.dtype)]}:chw"
    for out in graph.outputs:
        Output_format[out.name] = f"{dtype_dict[str(out.dtype)]}:chw"
    return Input_format, Output_format   

def show_trtexec_cmd(onnx_path, device=None, strongly=False, write2file=None):
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)
    origin_input_format, origin_output_format = get_onnx_IOformat(graph)

    Input_format = {}

    dynamic_shape_flag="--minShapes=ranks_depth:629294,ranks_feat:629294,ranks_bev:629294,interval_starts:8507,interval_lengths:8507 --optShapes=ranks_depth:1258588,ranks_feat:1258588,ranks_bev:1258588,interval_starts:17014,interval_lengths:17014 --maxShapes=ranks_depth:5034352,ranks_feat:5034352,ranks_bev:5034352,interval_starts:68056,interval_lengths:68056 " if 'ranks_bev' in graph.tensors().keys() else ""
    BEV_OD_layer_precision="--layerPrecisions=/BEV_OD/head1/layers.4/attn/MatMul_3:fp32,/BEV_OD/head1/layers.5/attn/MatMul_3:fp32,/BEV_OD/head1/layers.11/attn/MatMul_3:fp32,/BEV_OD/head1/layers.12/attn/MatMul_3:fp32,/BEV_OD/head1/layers.3/layers/layers.9/LayerNormalization:fp32,/BEV_OD/head1/layers.10/layers/layers.9/LayerNormalization:fp32,/BEV_OD/head1/layers.17/layers/layers.9/LayerNormalization:fp32 --precisionConstraints=prefer" if 'sparse4d_prev2cur' in graph.tensors().keys() else ""

    if device in ["thor", None]:
        print("=== Thor tensorrt 10.10 转模型命令 for TRT 10.* (7020 / 7030 / thor):")
        inp_IO_format = "--inputIOFormats="
        for inp in graph.inputs:
            if inp.name in thor_special_input_formats:
                inp_IO_format+=f"{thor_special_input_formats[inp.name]},"
            else:
                inp_IO_format+=f"{origin_input_format[inp.name]},"
        inp_IO_format=inp_IO_format[:-1]

        out_IO_format = "--outputIOFormats="
        for out in graph.outputs:
            if out.name in thor_special_output_formats:
                out_IO_format+=f"{thor_special_output_formats[out.name]},"
            else:
                out_IO_format+=f"{origin_output_format[out.name]},"
        out_IO_format=out_IO_format[:-1]
        if strongly:
            precision=f"--stronglyTyped"
        else:
            precision=f"--fp16 --int8 {BEV_OD_layer_precision}"
        thor_cmd = f"trtexec --useCudaGraph --useSpinWait --profilingVerbosity=detailed --verbose --disableHFusion --maxTactics=999 --maxAuxStreams=0 --tacticSources=-CUBLAS,-CUDNN,-CUBLASLT --separateProfileRun --noDataTransfers --noTF32 --enablePerfMode2 {precision} {inp_IO_format} {out_IO_format} {dynamic_shape_flag}--onnx={onnx_path} --staticPlugins=libinferplugin.so --saveEngine=model.trt"
        print(thor_cmd)
        if write2file:
            with open(write2file, "a") as f:
                f.write(f"\n{thor_cmd}\n")
    

    if device in ["orin",  "a800", None]:
        print("=== Orin tensorrt 8.6 转模型命令 for TRT 8.* (6090 / orin): ")
        inp_IO_format = "--inputIOFormats="
        for inp in graph.inputs:
            if inp.name in orin_special_input_formats:
                inp_IO_format+=f"{orin_special_input_formats[inp.name]},"
            else:
                inp_IO_format+=f"{origin_input_format[inp.name]},"
        inp_IO_format=inp_IO_format[:-1]

        out_IO_format = "--outputIOFormats="
        for out in graph.outputs:
            if out.name in orin_special_output_formats:
                out_IO_format+=f"{orin_special_output_formats[out.name]},"
            else:
                out_IO_format+=f"{origin_output_format[out.name]},"
        out_IO_format=out_IO_format[:-1]

        orin_cmd = f"trtexec --useCudaGraph --useSpinWait --profilingVerbosity=detailed --verbose --maxAuxStreams=0 --tacticSources=-CUBLAS,-CUDNN,-CUBLASLT --separateProfileRun --noDataTransfers --noTF32 --fp16 --int8 --sparsity=enable {BEV_OD_layer_precision} {inp_IO_format} {out_IO_format} {dynamic_shape_flag}--onnx={onnx_path} --staticPlugins=libinferplugin.so --saveEngine=model.trt"
        print(orin_cmd)
        if write2file:
            with open(write2file, "a") as f:
                f.write(f"\n{orin_cmd}\n")

    if device in ["dla", None]:
        print("=== dla 转模型命令 for TRT 8.* : ")
        inp_IO_format = "--inputIOFormats="
        for inp in graph.inputs:
            if inp.name in orin_special_input_formats:
                inp_IO_format+=f"{orin_special_input_formats[inp.name]},"
            else:
                inp_IO_format+=f"{origin_input_format[inp.name]},"
        inp_IO_format=inp_IO_format[:-1]

        out_IO_format = "--outputIOFormats="
        for out in graph.outputs:
            if out.name in orin_special_output_formats:
                out_IO_format+=f"{orin_special_output_formats[out.name]},"
            else:
                out_IO_format+=f"{origin_output_format[out.name]},"
        out_IO_format=out_IO_format[:-1]

        dla_cmd = f"trtexec --useCudaGraph --useSpinWait --profilingVerbosity=detailed --verbose --maxAuxStreams=0 --tacticSources=-CUBLAS,-CUDNN,-CUBLASLT --separateProfileRun --noDataTransfers --noTF32 --fp16 --int8 --useDLACore=0 --allowGPUFallback --sparsity=enable {inp_IO_format} --onnx={onnx_path} --calib={onnx_path.replace('.onnx','.calib')} --staticPlugins=libinferplugin.so --saveEngine=model.trt"
        print(dla_cmd)
        if write2file:
            with open(write2file, "a") as f:
                f.write(f"\n{dla_cmd}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='iso_the onnx model with new input and output')
    parser.add_argument('model', default="/media/models/vision+env/unified_QAT/TASK1111111_sim.onnx", type=str, help='the onnx model to deal with')
    args = parser.parse_args()
    # show_IO_formats(args.model)
    show_trtexec_cmd(args.model, device=None)