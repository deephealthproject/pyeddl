# Copyright (c) 2019-2020, CRS4
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
    license="BSD",
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    packages=["pyeddl"],
    ext_modules=[ext],
    install_requires=["setuptools", "pybind11", "numpy"],
    zip_safe=False,
)
