#!/bin/bash

set -euo pipefail

this="${BASH_SOURCE:-$0}"
this_dir=$(cd -P -- "$(dirname -- "${this}")" && pwd -P)

pushd "${this_dir}"
rm -rf include && mkdir include
docker run --rm eddl bash -c 'cat /usr/include/zlib.h' > include/zlib.h
docker run --rm eddl bash -c 'cat /usr/include/zconf.h' > include/zconf.h
pushd include
ln -s ../third_party/eddl/src eddl
popd
popd

pyeddl_mount=/pyeddl

docker run --rm -v "${this_dir}":"${pyeddl_mount}" \
  -e BINDER_EXE='/binder/build/llvm-5.0.0/build_5.0.0*/bin/binder' \
  -e EDDL_INCLUDE="${pyeddl_mount}"/include \
  -e EIGEN_INCLUDE="${pyeddl_mount}"/third_party/eddl/third_party/eigen \
  -w "${pyeddl_mount}"/codegen crs4/binder:135f6e3 ./gen_bindings.sh
cp codegen/bindings/_core.cpp src/
