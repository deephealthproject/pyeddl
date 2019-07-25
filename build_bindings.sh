#!/bin/bash

set -euo pipefail

# re-generate src/pyeddl.cpp

docker run --rm -v "${PWD}":/pyeddl -w /pyeddl/codegen crs4/binder:2f3665b ./gen_bindings.sh
cp codegen/bindings/pyeddl.cpp src/
