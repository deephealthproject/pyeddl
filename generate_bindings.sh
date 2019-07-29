#!/bin/bash

set -euo pipefail

this="${BASH_SOURCE:-$0}"
this_dir=$(cd -P -- "$(dirname -- "${this}")" && pwd -P)

pushd "${this_dir}"
rm -rf include && mkdir include
pushd include
ln -s ../third_party/eddl/src eddl
popd
popd

pyeddl_mount=/pyeddl

docker run --rm -v "${this_dir}":"${pyeddl_mount}" \
  -e BINDER_EXE='/binder/build/llvm-4.0.0/build_4.0.0*/bin/binder' \
  -e EDDL_INCLUDE="${pyeddl_mount}"/include \
  -e EIGEN_INCLUDE="${pyeddl_mount}"/third_party/eddl/third_party/eigen \
  -w "${pyeddl_mount}"/codegen crs4/binder:2f3665b ./gen_bindings.sh
cp codegen/bindings/_core.cpp src/
