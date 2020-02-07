#!/usr/bin/env bash

set -euo pipefail

this="${BASH_SOURCE-$0}"
this_dir=$(cd -P -- "$(dirname -- "${this}")" && pwd -P)

pushd "${this_dir}"
pushd third_party/eddl
eddl_rev=$(git rev-parse --short HEAD)
[ "${eddl_rev}" = "c023a6e" -o "${eddl_rev}" = "01ef395" ] && git apply ../../eddl_0.3.patch
popd
docker build -t simleo/eddl:${eddl_rev} -f Dockerfile.eddl .
docker build -f Dockerfile.jenkins --build-arg eddl_rev=${eddl_rev} -t simleo/pyeddl-base:${eddl_rev} .
docker build -t simleo/eddl-gpu:${eddl_rev} -f Dockerfile.eddl-gpu .
docker build -f Dockerfile.jenkins-gpu --build-arg eddl_rev=${eddl_rev} -t simleo/pyeddl-gpu-base:${eddl_rev} .
popd
