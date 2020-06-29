#!/bin/bash

ln -s /usr/include/cublas_v2.h "${PREFIX}"/include

mkdir build
cd build
cmake -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -DBUILD_PROTOBUF=ON \
      -DBUILD_TARGET=GPU \
      ${SRC_DIR}
make -j${CPU_COUNT}
make install

rm -fv "${PREFIX}"/include/cublas_v2.h
