import os
from distutils.core import setup, Extension

import pybind11


EXTRA_COMPILE_ARGS = ['-std=c++11', '-fvisibility=hidden']
LIBRARIES = ["eddl"]
if "EDDL_WITH_CUDA" in os.environ:
    LIBRARIES.extend(["cudart", "cublas", "curand"])


ext = Extension(
    "pyeddl._core",
    sources=["src/_core.cpp"],
    include_dirs=[
        "src",
        pybind11.get_include(),
        pybind11.get_include(user=True)
    ],
    libraries=LIBRARIES,
    extra_compile_args=EXTRA_COMPILE_ARGS,
)


setup(
    name="pyeddl",
    packages=["pyeddl"],
    ext_modules=[ext]
)
