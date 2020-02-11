#!/bin/bash

set -e

this="${BASH_SOURCE:-$0}"
this_dir=$(cd -P -- "$(dirname -- "${this}")" && pwd -P)

# this refers to the crs4/binder docker container
binder=${BINDER_EXE:-/binder/build/llvm-5.0.0/build_5.0.0*/bin/binder}

eddl_inc=${EDDL_INCLUDE:-"${this_dir}"/../third_party/eddl/src}
eigen_inc=${EIGEN_INCLUDE:-"${this_dir}"/../third_party/eddl/third_party/eigen}

rm -rf ./bindings/ && mkdir bindings/
${binder} \
  --root-module _core \
  --prefix $PWD/bindings/ \
  --bind eddl \
  --bind eddlT \
  --config config.cfg \
  --single-file \
  all_includes.hpp \
  -- -std=c++11 \
  -I"${eddl_inc}" \
  -I"${eigen_inc}" \
  -I"${this_dir}"/../src \
  -DNDEBUG

# add buffer_protocol annotation
sed -i -f add_annotation.sed bindings/_core.cpp

# set nodelete option
sed -i 's/shared_ptr<Metric/unique_ptr<Metric, pybind11::nodelete/' bindings/_core.cpp
