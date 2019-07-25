#!/bin/bash

set -e

# this should be used from the crs4/binder docker container
binder=/binder/build/llvm-4.0.0/build_4.0.0*/bin/binder

rm -rf ./bindings/ && mkdir bindings/
${binder} \
  --root-module pyeddl \
  --prefix $PWD/bindings/ \
  --bind "" \
  --config config.cfg \
  --single-file \
  all_bash_includes.hpp \
  -- -std=c++11 \
  -I${PWD}/../third_party/eddl/src \
  -I${PWD}/../third_party/eddl/third_party/eigen \
  -I${PWD}/../src \
  -DNDEBUG

# Fix for pybind11 ImportError
# "overloading a method with both static and instance methods is not supported"
sed -i 's/def("sum"/def("sum_unary"/' bindings/pyeddl.cpp

# add buffer_protocol annotation
sed -i -f add_annotation.sed bindings/pyeddl.cpp
