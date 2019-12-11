# Copyright (c) 2019 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
