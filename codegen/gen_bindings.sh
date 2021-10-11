#!/bin/bash

set -e

this="${BASH_SOURCE:-$0}"
this_dir=$(cd -P -- "$(dirname -- "${this}")" && pwd -P)

# this refers to the crs4/binder docker container
binder=${BINDER_EXE:-/binder/build/llvm-5.0.0/build_5.0.0*/bin/binder}

eddl_inc=${EDDL_INCLUDE:-"${this_dir}"/../third_party/eddl/include}

rm -rf ./bindings/ && mkdir bindings/
${binder} \
  --root-module _core \
  --prefix $PWD/bindings/ \
  --bind eddl \
  --config config.cfg \
  --single-file \
  all_includes.hpp \
  -- -std=c++11 \
  -I"${eddl_inc}" \
  -I"${this_dir}"/../src \
  -DNDEBUG

# add buffer_protocol annotation
sed -i -f add_annotation.sed bindings/_core.cpp

# set nodelete option
for c in Metric Loss Optimizer SGD Adam AdaDelta Adagrad Adamax Nadam RMSProp; do
    sed -i "s/shared_ptr<${c}>/unique_ptr<${c}, pybind11::nodelete>/" bindings/_core.cpp
done

# add custom binding section for utils
sed -i '/bind_eddl_descriptors_tensor_descriptors(M);/s/.*/&\
	bind_eddl_utils(M);/' bindings/_core.cpp
