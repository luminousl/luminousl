# import os
# import glob
# import torch
# from torch.utils.cpp_extension import CUDA_HOME
# from torch.utils.cpp_extension import CppExtension
# from torch.utils.cpp_extension import CUDAExtension
# from setuptools import find_packages
# from setuptools import setup

# requirements = ["torch", "torchvision"]

# def get_extensions():
#     this_dir = os.path.dirname(os.path.abspath(__file__))
#     extensions_dir = os.path.join(this_dir, "src")

#     extension = CppExtension
#     extra_compile_args = {"cxx": []}
#     define_macros = []

#     if torch.cuda.is_available() and CUDA_HOME is not None:
#         extension = CUDAExtension
#         define_macros += [("WITH_CUDA", None)]
#         extra_compile_args["nvcc"] = [
#             "-DCUDA_HAS_FP16=1",
#             "-D__CUDA_NO_HALF_OPERATORS__",
#             "-D__CUDA_NO_HALF_CONVERSIONS__",
#             "-D__CUDA_NO_HALF2_OPERATORS__",
#         ]
#     else:
#         raise NotImplementedError('Cuda is not availabel')

#     sources = ['src/bev_pool_v2.cpp', 'src/bev_pool_cuda_v2.cu']
#     include_dirs = [extensions_dir]
#     ext_modules = [
#         extension(
#             "bev_pool_v2_ext",
#             sources,
#             include_dirs=include_dirs,
#             define_macros=define_macros,
#             extra_compile_args=extra_compile_args,
#         )
#     ]
#     return ext_modules


# setup(
#     name="bev_pool_v2_ext",
#     version="1.0",
#     author="Le.Tai",
#     description="PyTorch Wrapper for bev_pool_v2",
#     packages=find_packages(exclude=("configs", "tests",)),
#     ext_modules=get_extensions(),
#     cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension}
# )



from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name='bev_pool_v2_ext',  # 扩展的名字
        sources=['src/bev_pool_v2.cpp', 'src/bev_pool_cuda_v2.cu'],  # 源文件列表
    )
]

setup(
    name='bev_pool_v2_ext',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
