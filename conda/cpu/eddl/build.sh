#!/bin/bash

mkdir build
cd build
cmake -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      ${SRC_DIR}
make -j${CPU_COUNT}
make install
