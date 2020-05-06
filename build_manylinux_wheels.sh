#!/usr/bin/env bash

set -euo pipefail

docker build -t eddl-manylinux -f Dockerfile.eddl-manylinux .
docker build -t pyeddl-manylinux -f Dockerfile.manylinux .
docker run --rm pyeddl-manylinux bash -c "tar -c -C /pyeddl wheels" | tar -x -C /tmp
