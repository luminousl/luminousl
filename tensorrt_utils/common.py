#!/usr/bin/env python
# coding: utf-8
# %%

import os
import numpy as np
import tensorrt as trt
import struct


dtype_map = {
    trt.tensorrt.float32: np.float32,
    trt.tensorrt.float16: np.float16,
    trt.tensorrt.int32: np.int32,
    trt.tensorrt.int8: np.int8,
    trt.tensorrt.bool: bool,
}

def _FakeDataGenerator(shape, directory, num):
    """
    Generate 'num' random numpy arrays of the specified 'shape', 
    and save them in the 'directory' with filenames 1.bin, 2.bin, ..., num.bin.

    :param shape: Tuple defining the shape of each numpy array.
    :param directory: Directory where the files will be saved.
    :param num: Number of numpy arrays (and files) to generate.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(1, num + 1):
        data =10 * np.random.rand(*shape)  # Generate random data
#         filename = os.path.join(directory, f"{i}.bin")  # Generate filename
#         data.tofile(filename)  # Save data to file in binary format
        filename = os.path.join(directory, f"{i}.npy")
        np.save(filename, data)

def FakeDataGenerator(directory, num, inp_info):
    """
    Generate 'num' random numpy arrays of the specified 'shape', 
    and save them in the 'directory' with filenames 1.bin, 2.bin, ..., num.bin.

    :param shape: Tuple defining the shape of each numpy array.
    :param directory: Directory where the files will be saved.
    :param num: Number of numpy arrays (and files) to generate.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(1, num + 1):
        data_dict = {}
        for name, info in inp_info.items():
            data =np.random.rand(*(info['shape']))  # Generate random data
            # To do INT, np.random.randint(0, 10, size=(3, 3), dtype=np.int32)
            # to do bool.
            data = data.astype(info['dtype'])
            data_dict[name]=data
        filename = os.path.join(directory, f"{i}.npy")
        np.save(filename, data_dict)

def get_input_info(**kwargs):
    network = kwargs.get('network', None)
    engine = kwargs.get('engine', None)
    input_info = {}
    if network:
        for ind in range(network.num_inputs):
            inp = network.get_input(ind)
            info = {'shape': inp.shape,
            'dtype': dtype_map[inp.dtype]}
            input_info[inp.name] = info
        return input_info
    elif engine:
        for inp_name in engine:
            if engine.get_tensor_mode(inp_name) == trt.TensorIOMode.INPUT:
                info = {'shape': engine.get_tensor_shape(inp_name),
                'dtype': dtype_map[engine.get_tensor_dtype(inp_name)]}
                input_info[inp_name] = info
        return input_info
    else:
        pass
    return None


def get_batch_sizes(max_batch_size):
    # Returns powers of 2, up to and including max_batch_size
    max_exponent = math.log2(max_batch_size)
    for i in range(int(max_exponent)+1):
        batch_size = 2**i
        yield batch_size
    
    if max_batch_size != batch_size:
        yield max_batch_size


# TODO: This only covers dynamic shape for batch size, not dynamic shape for other dimensions
def create_optimization_profiles(builder, inputs, batch_sizes=[1,8,16,32,64]): 
    # Check if all inputs are fixed explicit batch to create a single profile and avoid duplicates
    if all([inp.shape[0] > -1 for inp in inputs]):
        profile = builder.create_optimization_profile()
        for inp in inputs:
            fbs, shape = inp.shape[0], inp.shape[1:]
            profile.set_shape(inp.name, min=(fbs, *shape), opt=(fbs, *shape), max=(fbs, *shape))
            return [profile]
    
    # Otherwise for mixed fixed+dynamic explicit batch inputs, create several profiles
    profiles = {}
    for bs in batch_sizes:
        if not profiles.get(bs):
            profiles[bs] = builder.create_optimization_profile()

        for inp in inputs: 
            shape = inp.shape[1:]
            # Check if fixed explicit batch
            if inp.shape[0] > -1:
                bs = inp.shape[0]

            profiles[bs].set_shape(inp.name, min=(bs, *shape), opt=(bs, *shape), max=(bs, *shape))

    return list(profiles.values())

def get_tensor_by_name(network, tensor_name):
    for layer in network:
        for output_index in range(layer.num_outputs):
            if layer.get_output(output_index).name == tensor_name:
                return layer.get_output(output_index)
    return None

def read_calibtable_txt2json(calib_file):
    results = []
    with open(calib_file) as calibtxt:
        for line in calibtxt:
            if ":" not in line:
                continue
            line = line.strip().split(':')
            result = struct.unpack('>f', bytes.fromhex(line[1]))[0]
            results.append(result)
            print(struct.unpack('>f', bytes.fromhex(line[1]))[0])
#             float_val = struct.unpack('>f', bytes.fromhex(float_hex))[0]
    return results

# %%
if __name__ == '__main__':
    pass