#!/usr/bin/env bash

set -euo pipefail

docker build -t manylinux-cuda101 -f Dockerfile.manylinux-cuda101 .
docker build -t eddl-manylinux-gpu -f Dockerfile.eddl-manylinux-gpu .
docker build -t pyeddl-manylinux-gpu -f Dockerfile.manylinux-gpu .
# copy the wheels to /tmp/wheels on the host
docker run --rm pyeddl-manylinux-gpu bash -c "tar -c -C /pyeddl wheels" | tar -x -C /tmp
