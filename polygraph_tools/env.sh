export CUDA_HOME=/usr/local/cuda-11.3  # 修改项
export TRT_HOME=/mnt/polygraphy_test/TensorRT-10.11.0.33.Linux.x86_64-gnu.cuda-11.8/TensorRT-10.11.0.33  # 修改项

# # 修正LD_LIBRARY_PATH顺序，TensorRT应该在前
export LD_LIBRARY_PATH=$TRT_HOME/lib:$LD_LIBRARY_PATH
# 添加Python路径（重要！）
export PYTHONPATH=$TRT_HOME/python:$PYTHONPATH