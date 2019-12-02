#!/usr/bin/env bash

# Assumes EDDL compiled for GPU.

set -euo pipefail
[ -n "${DEBUG:-}" ] && set -x
this="${BASH_SOURCE-$0}"
this_dir=$(cd -P -- "$(dirname -- "${this}")" && pwd -P)

names=(
    eddl_ae
    eddl_conv
    eddl_dae_class
    eddl_load_save
    eddl_mlp
    eddl_train_batch
)

for n in "${names[@]}"; do
    echo -en "\n*** ${n} ***\n"
    python3 "${this_dir}"/${n}.py --gpu --epochs 1
done
