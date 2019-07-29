#!/bin/bash

set -euo pipefail

pyinc=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')

g++ \
    -O3 \
    -I"${pyinc}" \
    -I${PWD}/third_party/eddl/third_party/pybind11/include \
    -I${PWD}/src \
    -shared  \
    -std=c++11 \
    -c src/pyeddl.cpp  \
    -o pyeddl.o \
    -fPIC

g++ -o pyeddl.so -shared pyeddl.o -leddl -lstdc++ -Wl,-rpath
