#!/usr/bin/env bash

set -euo pipefail

docker build -t eddl-superbuild -f Dockerfile.eddl-superbuild .
docker build -t pyeddl-superbuild -f Dockerfile.superbuild .
