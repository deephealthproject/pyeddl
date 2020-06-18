#!/usr/bin/env bash

set -euo pipefail

pkg_dir="dhealth/linux-64"

pkg=$(docker run --rm pyeddl-conda bash -c 'ls -1 /opt/conda/conda-bld/linux-64/pyeddl-cpu-* | head -n 1')
docker run --rm pyeddl-conda bash -c "cat ${pkg}" > "${pkg_dir}/$(basename ${pkg})"
