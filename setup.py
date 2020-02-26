# Copyright (c) 2019-2020 CRS4
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

"""\
PyEDDL is a Python wrapper for EDDL, the European Distributed Deep
Learning library.
"""

import os
from setuptools import setup, Extension

import pybind11
from pyeddl.version import VERSION


def to_bool(s):
    s = s.lower()
    return s != "off" and s != "false"


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

# optional modules, on by default. Set env var to "OFF" or "FALSE" to disable
EDDL_WITH_PROTOBUF = to_bool(os.getenv("EDDL_WITH_PROTOBUF", "ON"))
if EDDL_WITH_PROTOBUF:
    EXTRA_COMPILE_ARGS.append('-DEDDL_WITH_PROTOBUF')


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
    version=VERSION,
    url="https://github.com/deephealthproject/pyeddl",
    description="Python wrapper for EDDL",
    long_description=__doc__,
    author="Simone Leo",
    author_email="<simone.leo@crs4.it>",
    license="MIT",
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    packages=["pyeddl"],
    ext_modules=[ext],
    install_requires=["setuptools", "pybind11", "numpy"],
    zip_safe=False,
)
