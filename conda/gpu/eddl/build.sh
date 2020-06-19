#!/bin/bash

# for nvcc
gcc_exe=$(find "${BUILD_PREFIX}"/bin/ -name '*gcc' -type f | head -n 1)
ln -s "${gcc_exe}" "${PREFIX}"/bin/gcc

mkdir build
cd build
cmake -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -DBUILD_PROTOBUF=ON \
      -DBUILD_TARGET=GPU \
      -DCMAKE_CUDA_COMPILER=${PREFIX}/bin/nvcc \
      -DCMAKE_CUDA_COMPILER_FORCED=${PREFIX}/bin/nvcc \
      -DCMAKE_CUDA_FLAGS="--compiler-options -fPIC" \
      ${SRC_DIR}
make -j${CPU_COUNT}
make install

rm -fv "${PREFIX}"/bin/gcc
