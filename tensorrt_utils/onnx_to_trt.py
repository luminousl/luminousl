#!/usr/bin/env python3

# +
import os
import sys
import glob
import math
import logging
import argparse
import struct
import tensorrt as trt
import yaml

# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#                     datefmt="%Y-%m-%d %H:%M:%S")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
# logging.getLogger('matplotlib').setLevel(logging.INFO)
# logging.getLogger('graphviz').setLevel(logging.INFO)

sys.path.append('/media/Projects/MATool/utils')
sys.path.append('/media/Projects/MATool')

from MultiInputCalibrator import FakeInputCalibrator, get_calibration_files, get_int8_calibrator
from process_onnx import onnx2trt
from trex import *
from process_trt import trt2profile

TRT_LOGGER = trt.Logger()
logger = logging.getLogger(__name__)

from common import *

def add_profiles(config, inputs, opt_profiles):
    logger.info("=== Optimization Profiles ===")
    for i, profile in enumerate(opt_profiles):
        for inp in inputs:
            _min, _opt, _max = profile.get_shape(inp.name)
            logger.debug("{} - OptProfile {} - Min {} Opt {} Max {}".format(inp.name, i, _min, _opt, _max))
        config.add_optimization_profile(profile)

def mark_outputs(network, tensor_name_list):
    # Mark network outputs by tensor name
    logger.info(f">>> Marking Network output")
    for name in tensor_name_list:
        tensor = get_tensor_by_name(network, name)
        if tensor:
            network.mark_output(tensor)
            logger.info(f"name: {tensor.name}, shape: {tensor.shape}")
        else:
            logger.error(f"Tensor {name} marked not found")
    

def check_network(network):
    logger.info(">>> Network Checking")

    if not network.num_outputs:
        logger.warning("No output nodes found, marking last layer's outputs as network outputs. Correct this if wrong.")
        mark_outputs(network)
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    max_len = max([len(inp.name) for inp in inputs] + [len(out.name) for out in outputs])

    for i, inp in enumerate(inputs):
        logger.info("Input  {0} | Name: {1:{2}} | Shape: {3}".format(i, inp.name, max_len, inp.shape))
    for i, out in enumerate(outputs):
        logger.info("Output {0} | Name: {1:{2}} | Shape: {3}".format(i, out.name, max_len, out.shape))

def get_network_flags(args):
    network_flags = 0
    if args.explicit_batch:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        # print("explicit_batch: ", network_flags)
    return network_flags


def set_logger(args):
    # Adjust logging verbosity
    if args.verbosity is None:
        TRT_LOGGER.min_severity = trt.Logger.Severity.ERROR
    # -v
    elif args.verbosity == 1:
        TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
    # -vv
    else:
        TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE

    logger.info("TRT_LOGGER Verbosity: {:}".format(TRT_LOGGER.min_severity))

def set_network(args, network, output_name_to_mark):
    logger.info("=== SETTING NETWORK ===")

    parser = trt.OnnxParser(network, TRT_LOGGER)
    # Fill network atrributes with information by parsing model
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            print('ERROR: Failed to parse the ONNX file: {}'.format(args.onnx))
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    mark_outputs(network, output_name_to_mark)
    check_network(network)

def set_config(args, builder, config, network):
    logger.info("=== SETTING CONFIG ===")
    builder_flag_map = {
            'gpu_fallback': trt.BuilderFlag.GPU_FALLBACK,
            'refittable': trt.BuilderFlag.REFIT,
            'debug': trt.BuilderFlag.DEBUG,
            # 'strict_types': trt.BuilderFlag.STRICT_TYPES,
            'fp16': trt.BuilderFlag.FP16,
            'int8': trt.BuilderFlag.INT8,
            'prefer_precision_constraints': trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS,
    }
    

    # Set Builder Config Flags
    for flag in builder_flag_map:
        if getattr(args, flag):
            logger.info("Config Setting {}".format(builder_flag_map[flag]))
            config.set_flag(builder_flag_map[flag])
            
    # config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    # logger.info("Config Setting {}".format(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS))

    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    logger.info("Config Setting Verbosity={}".format(config.profiling_verbosity))

    if args.fp16 and not builder.platform_has_fast_fp16:
        logger.warning("FP16 not supported on this platform.")

    if args.int8 and not builder.platform_has_fast_int8:
        logger.warning("INT8 not supported on this platform.")


    if args.int8:
        logger.info("Use Int8 setting")
        logger.info("Prepare int8 calibrator")
        # input_shape = (1, 3, 224, 224)
        input_info = get_input_info(network=network)
        # Fake data generation
        if args.fake_data:
            logger.info(f"Fake data directory: {args.fake_data}")
            if os.path.exists(args.fake_data):
                logger.info("Using existing data for calibration")
        #         raise Exception("Fake Data Directory already exists")
            else:
                logger.info("Generate Fake data for calibration")
                FakeDataGenerator(args.fake_data, args.max_calibration_size, input_info)
        cali_data = args.fake_data if args.fake_data else args.calibration_data

        config.int8_calibrator = get_int8_calibrator(args.calibration_cache,
                                                     cali_data,
                                                     args.max_calibration_size,
                                                     args.preprocess_func,
                                                     args.calibration_batch_size,
                                                     input_info)


def build_engine(args, builder, network, config, saveEngine=True, engine_path=None):
    # if args.explicit_batch:
    #     # Add optimization profiles
    #     batch_sizes = [1, 8, 16, 32, 64]
    #     inputs = [network.get_input(i) for i in range(network.num_inputs)]
    #     opt_profiles = create_optimization_profiles(builder, inputs, batch_sizes)
    #     add_profiles(config, inputs, opt_profiles)
    # # Implicit Batch Network
    # else:
    #     builder.max_batch_size = args.max_batch_size
    # builder.max_batch_size = 16
    logger.info("=== Engine Building ===")
    engine = builder.build_serialized_network(network, config)

    if saveEngine:
        engine_file_path = engine_path if engine_path else args.output
        with open(engine_file_path, "wb") as f:
            logger.info("Serializing engine to file: {:}".format(engine_file_path))
            f.write(engine)

    return engine

def profile_engine(engine_path, profileEngine=True):
    logger.info("=== Engine Profiling ===")
    if profileEngine:
        trt2profile(engine_path, ['best'])

def set_args_by_yaml(args, yaml_file, key='default'):
    if yaml_file:
        with open(yaml_file, 'r') as f:
            yaml_setting = yaml.load(f, Loader=yaml.FullLoader)[key]
        for k,v in yaml_setting.items():
            setattr(args, k, v)
    else:
        pass

def print_args(args, logger=None):
    if logger:
        # logger.setLevel(logging.INFO)
        message = 'Basic setting:\n' + '\n'.join([f'\t{k:<30}: {v}' for k, v in vars(args).items()])
        logger.info(message)
    else:
        for arg in vars(args):
            print(format(arg, '<30'), format(str(getattr(args, arg)), '<'))   # str, arg_type


def get_args(yaml_file=None, yaml_config_key=None):
    parser = argparse.ArgumentParser(description="Creates a TensorRT engine from the provided ONNX file.\n")
    parser.add_argument("-y", "--yaml", type=str, default=None, help="Setting config from yaml file")
    parser.add_argument("-k", "--yaml_config_key", type=str, default=None, help="The config key in the yaml file")
    parser.add_argument("--onnx", type=str, default='./ResNet50.onnx', help="The ONNX model file to convert to TensorRT")
    parser.add_argument("-o", "--output", type=str, default="ResNet50.engine", help="The path at which to write the engine")
    # parser.add_argument("-b", "--max-batch-size", type=int, default=1, help="The max batch size for the TensorRT engine input")
    parser.add_argument("-v", "--verbosity", action="count", help="Verbosity for logging. (None) for ERROR, (-v) for INFO/WARNING/ERROR, (-vv) for VERBOSE.")
    parser.add_argument("--explicit-batch", action='store_true', help="Set trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH.")
    parser.add_argument("--gpu-fallback", action='store_true', help="Set trt.BuilderFlag.GPU_FALLBACK.")
    parser.add_argument("--refittable", action='store_true', help="Set trt.BuilderFlag.REFIT.")
    parser.add_argument("--debug", action='store_true', help="Set trt.BuilderFlag.DEBUG.")
    # parser.add_argument("--strict-types", action='store_true', help="Set trt.BuilderFlag.STRICT_TYPES.")
    parser.add_argument("--prefer_precision_constraints", action='store_true', help="Set trt.BuilderFlag.STRICT_TYPES.")
    parser.add_argument("--fp16", action="store_true", help="Attempt to use FP16 kernels when possible.")
    parser.add_argument("--int8", action="store_true", help="Attempt to use INT8 kernels when possible. This should generally be used in addition to the --fp16 flag. \
                                                             ONLY SUPPORTS RESNET-LIKE MODELS SUCH AS RESNET50/VGG16/INCEPTION/etc.")
    parser.add_argument("--calibration-cache", help="(INT8 ONLY) The path to read/write from calibration cache.", default="calibration.cache")
    parser.add_argument("--calibration-data", help="(INT8 ONLY) The directory containing {*.jpg, *.jpeg, *.png} files to use for calibration. (ex: Imagenet Validation Set)", default='images')
    parser.add_argument("--calibration-batch-size", help="(INT8 ONLY) The batch size to use during calibration.", type=int, default=1)
    parser.add_argument("--max-calibration-size", help="(INT8 ONLY) The max number of data to calibrate on from --calibration-data.", type=int, default=50)
    parser.add_argument("-p", "--preprocess_func", type=str, default=None, help="(INT8 ONLY) Function defined in 'processing.py' to use for pre-processing calibration data.")
    parser.add_argument("-f", "--fake-data", type=str, default=None, help="Whether to generate Fake Calibration data")
    args, _ = parser.parse_known_args()

    args.yaml = yaml_file if yaml_file else args.yaml
    args.yaml_config_key = yaml_config_key if yaml_config_key else args.yaml_config_key
    
    set_args_by_yaml(args, args.yaml, key=args.yaml_config_key)
    print_args(args, logger=logger)    
    # args.onnx = "ResNet50.onnx"
    # args.output = "ResNet50.engine"
    # args.onnx = "/media/Projects/ModelTest/PTQ_calibration_generation/resnet18.onnx"
    # args.output = "/media/Projects/ModelToolbox/trt_utils/tmp/resnet_18_engine/fp32.trt"
    # args.onnx = "/media/models/Resnet34_3inputs_448x448_20200609/Resnet34_3inputs_448x448_20200609.onnx"
    # args.output = "/media/Projects/ModelToolbox/trt_utils/tmp/Resnet34_3inputs_448x448_20200609.engine"
    # args.int8 = True
    # args.fp16 = True
    # args.calibration_cache = "/media/Projects/ModelToolbox/trt_utils/tmp/engine18.cali"
    # args.fake_data = "/media/Projects/ModelToolbox/trt_utils/tmp/Fake_cali_data_18"
    # args.explicit_batch = True
    # args.verbosity = 1
    # args.strict_types = True
    # args.prefer_precision_constraints = True

    return args


def main(args):
    # args = get_args()
    set_logger(args)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    network = builder.create_network(get_network_flags(args))

    output_name_to_mark = []
    set_network(args, network, output_name_to_mark)
    set_config(args, builder, config, network)

    ### config modification
    # config.int8_calibrator.modify_calibration_cache(['/layer2/layer2.0/conv2/Conv_output_0'])

    # config.int8_calibrator.modify_calibration_cache([
    #     '/layer2/layer2.0/conv2/Conv_output_0',
    #     '/layer2/layer2.0/downsample/downsample.0/Conv_output_0',
    #     '/layer2/layer2.0/relu/Relu_output_0'])
    

    build_engine(args, builder, network, config, saveEngine=True)
    profile_engine(args.output, profileEngine=True)

if __name__ == "__main__":
    yaml_file = 'QuantAnalysis/default.yaml'
    yaml_config_key = 'default'
    args = get_args(yaml_file=yaml_file, yaml_config_key=yaml_config_key)
    # main()