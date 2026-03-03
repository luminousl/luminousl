#!/usr/bin/env python
# coding: utf-8
# %%
import sys
sys.path.append('/media/Projects/ModelToolbox')
from trt_utils.onnx2trt import *
from trt_utils.common import *
from trt_utils.trt_excution import *
from trt_utils.MultiInputDataLoader import *
from math_utils.common import *


yaml_file = 'QuantAnalysis/default.yaml'
yaml_config_key = 'resnet18'
args = get_args(yaml_file, yaml_config_key)
set_logger(args)
builder = trt.Builder(TRT_LOGGER)
config = builder.create_builder_config()
network = builder.create_network(get_network_flags(args))

output_name_to_mark = []
set_network(args, network, output_name_to_mark)
set_config(args, builder, config, network)

# %%
# for ind in range(len(network)):
for ind in range(2):
    if network[ind].type==trt.tensorrt.LayerType.CONVOLUTION:
        network[ind].precision = trt.tensorrt.DataType.FLOAT
        engine_path = f'/media/Projects/ModelToolbox/trt_utils/tmp/PTQ_test_resnet18/{ind}.trt'
        print(f"Building engine with layer {network[ind].name} remain fp32")
        build_engine(args, builder, network, config, 
                     saveEngine=True, engine_path=engine_path)
        profile_engine(engine_path, profileEngine=True)
        network[ind].reset_precision()

# # %%
# profile_engine('/media/Projects/ModelToolbox/trt_utils/tmp/resnet_18_engine/0.trt', profileEngine=True)

# # %%
# engine = build_engine(args, builder, network, config, saveEngine=False)

# # %%
# # model = TrtModel(engine)
# model = TrtModel('/media/Projects/ModelToolbox/trt_utils/tmp/resnet18_PTQ.trt')
# # inputs = generate_random_input(model.engine)
# # result = model(inputs)
# # for k,v in result.items():
# #     print(k, v.shape)

# # %%
# data_loader = get_data_loader('/media/Projects/ModelToolbox/trt_utils/tmp/Fake_cali_data_18',
#                              100, None, get_input_info(engine=model.engine), 1)

# # %%
# A=data_loader.get_batch(['input.1'])

# # %%
# A[0].shape

# # %%
# A = read_calibtable_txt2json('/media/Projects/ModelToolbox/trt_utils/tmp/engine18.cali')
# B = read_calibtable_txt2json('/media/Projects/ModelToolbox/trt_utils/tmp/engine18 (copy).cali')
