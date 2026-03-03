from onnxconverter_common import float16
from utils.degroup import degroup
from utils.common import *
import argparse


def convert2strongly(ipath, opath):
    print(f"Convert {ipath} to FP16 model {opath}")
    node_block_list = []
    op_block_list=[]

    model = onnx.load(ipath)
    print("===  [s] Model from weakly To strongly ===")
    model = fix_ConvTransposeArgmax_mode(model)
    model_s = shrink_crazy_bn(model)
    model_s = float16.convert_float_to_float16(model_s, keep_io_types=True, disable_shape_infer=True, 
                                                 node_block_list=node_block_list, op_block_list=op_block_list, max_finite_val=np.finfo("float16").max)
    model_s = fixup_resize_scales_dtype(model_s)
    

    print("=== [d] replace group conv wight non-group conv ===")
    model_s = degroup(model_s)

    # plugin_add_module(model_s_d_p_c)
    onnx.save(model_s, opath)
    print(f"Modified model saved to {opath}")

    # 可选：检查模型有效性
    onnx.checker.check_model(opath)
    print("Model check passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="onnx file to be dealt with")
    args = parser.parse_args()

    input_onnx = args.input
    output_onnx = args.input.replace(".onnx", ".strongly.onnx")
    convert2strongly(input_onnx, output_onnx)
