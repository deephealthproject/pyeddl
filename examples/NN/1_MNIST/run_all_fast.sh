#!/usr/bin/env bash

# Assumes EDDL compiled for GPU.

set -euo pipefail
[ -n "${DEBUG:-}" ] && set -x
this="${BASH_SOURCE-$0}"
this_dir=$(cd -P -- "$(dirname -- "${this}")" && pwd -P)

names=(
    mnist_auto_encoder
    mnist_conv
    mnist_conv1D
    mnist_losses
    mnist_mlp_initializers
    mnist_mlp
    mnist_mlp_regularizers
    mnist_mlp_train_batch
    mnist_rnn
)

for n in "${names[@]}"; do
    echo -en "\n*** ${n} ***\n"
    python3 "${this_dir}"/${n}.py --gpu --epochs 1 --small
done

echo -en "\n*** mnist_mlp_da ***\n"
python3 "${this_dir}"/mnist_mlp_da.py --gpu --epochs 5 --small
