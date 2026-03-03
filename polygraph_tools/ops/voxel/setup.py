import os
import glob
import torch
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]


def make_cuda_ext(
    name, module, sources, sources_cuda=[], extra_args=[], extra_include_path=[]
):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        if "H800" in torch.cuda.get_device_name():
            extra_args.append("-gencode=arch=compute_90,code=sm_90")
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
        ]
        sources += sources_cuda
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )

setup(
    name="Voxelization",
    description="PyTorch Wrapper for CUDA Functions of point cloud voxelization",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=[make_cuda_ext(
                        name="voxel_layer",
                        module="",
                        sources=[
                            "src/voxelization.cpp",
                            "src/scatter_points_cpu.cpp",
                            "src/scatter_points_cuda.cu",
                            "src/voxelization_cpu.cpp",
                            "src/voxelization_cuda.cu",
                        ],
                    ),
                ],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    zip_safe=False,
)
