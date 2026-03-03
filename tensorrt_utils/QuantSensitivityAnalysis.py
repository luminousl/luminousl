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
from copy import deepcopy

args = get_args()
set_logger(args)
builder = trt.Builder(TRT_LOGGER)
config = builder.create_builder_config()
network = builder.create_network(get_network_flags(args))
set_network(args, network, output_name_to_mark=[])
# set_config(args, builder, config, network)

# %%
def test_model_result(model):
    data_loader = get_data_loader('/media/Projects/ModelToolbox/trt_utils/tmp/Fake_cali_data_18',
                                 100, None, get_input_info(network=network), 1)
    result = []
    while True:
        X=data_loader.get_batch(['input.1'])
        if X:
            result.append(deepcopy(model(X)))
        else:
            break
    return result


# %%
results = []
engine_path_list = []
engine_path = f'/media/Projects/ModelToolbox/trt_utils/tmp/resnet_18_engine/fp32.trt'
engine_path_list.append(engine_path)
model = TrtModel(engine_path)
results.append(test_model_result(model))
for ind in range(len(network)):
    if network[ind].type==trt.tensorrt.LayerType.CONVOLUTION:
        engine_path = f'/media/Projects/ModelToolbox/trt_utils/tmp/resnet_18_engine/{ind}.trt'
        engine_path_list.append(engine_path)
        model = TrtModel(engine_path)
        results.append(test_model_result(model))

# %%

# %%
metrics = []
source = results[0]
for target in results:
    similarity = [cosine_similarity(source[i]['191'], target[i]['191']) 
                  for i in range(50)]
    metrics.append(similarity)

# %%
metrics = np.array(metrics)

# %%
metrics.var(axis=1)

# %%
for i in np.argsort(metrics.var(axis=1)):
    print(engine_path_list[i])

# %%
