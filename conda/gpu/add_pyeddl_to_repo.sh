#!/usr/bin/env bash

set -euo pipefail

pkg_dir="dhealth/linux-64"

for v in 36 37 38; do
    pkg=$(docker run --rm pyeddl-conda bash -c "ls -1 /opt/conda/conda-bld/linux-64/pyeddl-gpu-*-py${v}* | head -n 1")
    echo "adding ${pkg}"
    docker run --rm pyeddl-conda bash -c "cat ${pkg}" > "${pkg_dir}/$(basename ${pkg})"
done

pyeddl_v=$(ls -1 "${pkg_dir}" | grep -m1 pyeddl | cut -d '-' -f 3)
echo downloading tests for pyeddl ${pyeddl_v}
wget -q https://github.com/deephealthproject/pyeddl/archive/${pyeddl_v}.tar.gz
tar xf ${pyeddl_v}.tar.gz
rm -rf tests
mv pyeddl-${pyeddl_v}/tests .
rm -rf pyeddl-${pyeddl_v}
