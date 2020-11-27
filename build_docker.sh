#!/usr/bin/env bash

set -euo pipefail

docker build -t eddl -f Dockerfile.eddl .
docker build -t pyeddl .
docker build -t pyeddl-docs -f Dockerfile.docs .
rm -rf /tmp/html && docker run --rm pyeddl-docs bash -c "tar -c -C /pyeddl/docs/source/_build html" | tar -x -C /tmp
