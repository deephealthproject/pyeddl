import os
from distutils.core import setup, Extension

import pybind11


EXTRA_COMPILE_ARGS = ['-std=c++11', '-fvisibility=hidden']
LIBRARIES = ["eddl"]
if "EDDL_WITH_CUDA" in os.environ:
    LIBRARIES.extend(["cudart", "cublas", "curand"])
INCLUDE_DIRS = [
    "src",
    pybind11.get_include(),
    pybind11.get_include(user=True)
]
LIBRARY_DIRS = []
RUNTIME_LIBRARY_DIRS = []
EDDL_DIR = os.getenv("EDDL_DIR")
if EDDL_DIR:
    INCLUDE_DIRS.extend([os.path.join(EDDL_DIR, "include")])
    LIBRARY_DIRS.extend([os.path.join(EDDL_DIR, "lib")])
    RUNTIME_LIBRARY_DIRS.extend([os.path.join(EDDL_DIR, "lib")])


ext = Extension(
    "pyeddl._core",
    sources=["src/_core.cpp"],
    include_dirs=INCLUDE_DIRS,
    library_dirs=LIBRARY_DIRS,
    runtime_library_dirs=RUNTIME_LIBRARY_DIRS,
    libraries=LIBRARIES,
    extra_compile_args=EXTRA_COMPILE_ARGS,
)


setup(
    name="pyeddl",
    packages=["pyeddl"],
    ext_modules=[ext]
)
