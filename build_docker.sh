#!/usr/bin/env bash

set -euo pipefail

docker build -t eddl -f Dockerfile.eddl .
docker build -t pyeddl .
