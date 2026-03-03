#!/usr/bin/env python
# coding: utf-8

import tensorrt as trt
import numpy as np
import os

import pycuda.driver as cuda
import pycuda.autoinit
from common import get_input_info


class HostDeviceMem(object):
    def __init__(self, name, host_mem, device_mem, shape):
        self.host = host_mem
        self.device = device_mem
        self.shape = shape
        self.name = name

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine,max_batch_size=1,dtype=np.float32):
        
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.INFO)
        # Create a runtime for TensorRT
        self.runtime = trt.Runtime(self.logger)
        self.logger.log(self.logger.INFO, "Loading TRT engine...")
        self.engine = self.load_engine(self.runtime, engine)
        # self.input_info = get_input_info(engine=self.engine)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()
    
    @staticmethod
    def load_engine(trt_runtime, engine):
        trt.init_libnvinfer_plugins(None, "")
        if isinstance(engine, str):
            with open(engine, 'rb') as f:
                engine_data = f.read()
        else: 
            engine_data = engine
        loaded_engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return loaded_engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(binding, host_mem, device_mem, shape))
            else:
                outputs.append(HostDeviceMem(binding, host_mem, device_mem, shape))
        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self,xs:list,batch_size=1):
        
        assert len(xs) == len(self.inputs)
        for i in range(len(xs)):
            x = xs[i].astype(self.dtype)
            np.copyto(self.inputs[i].host,x.ravel())
        
        for inp in self.inputs:
            # Transfer input data to device
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
            
        # Run inference
        # self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            # Transfer output data back to host
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        # Synchronize to wait for the inference to complete
        self.stream.synchronize()
        return {out.name: out.host.reshape(*out.shape) for out in self.outputs}

def generate_random_input(engine):
    inputs = []
    for binding in engine:
        if engine.binding_is_input(binding):
            shape = engine.get_binding_shape(binding)
            data = np.random.rand(*shape)
#             data = np.ones(shape)
            inputs.append(data)
    return inputs

   

# +
if __name__ == "__main__":

#     trt_engine_path = "/media/Projects/ModelTest/PTQ_calibration_generation/resnet18_int8.engine"
    trt_engine_path = "/media/Projects/ModelToolbox/trt_utils/tmp/Resnet34_3inputs_448x448_20200609.engine"
    model = TrtModel(trt_engine_path)
    
    inputs = generate_random_input(model.engine)
    result = model(inputs)
