
###########################################
# WORKDIR /home/nvidia/jw/profiling_debug/{TIME_NOW}
# HOSTIP 10.19.225.220
# HOSTUSER nvidia
# HOSTPASSWD nvidia

# LAYER_INFO_JSON outputs/layer_info.json
# PROFILE_JSON outputs/profile.json
# BUILD_LOG outputs/build.log
# CLEANUP True
# TREX_VIEW outputs/layer_info.json outputs/profile.json
###########################################

export LD_LIBRARY_PATH=/home/nvidia/jw/TensorRT-10.10.10.1/lib:${LD_LIBRARY_PATH}

/home/nvidia/jw/TensorRT-10.10.10.1/bin/trtexec --onnx=inputs/model.onnx --profilingVerbosity=detailed --fp16 --int8 --separateProfileRun \
    --exportLayerInfo=outputs/layer_info.json --verbose --exportProfile=outputs/profile.json > outputs/build.log 2>&1

# /home/nvidia/jw/TensorRT-10.10.10.1/bin/trtexec --help > outputs/trtversion 2>&1
# echo "hello world" > outputs/demo.txt