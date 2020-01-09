#!/usr/bin/env bash

# Assumes EDDL compiled for GPU.

set -euo pipefail
[ -n "${DEBUG:-}" ] && set -x
this="${BASH_SOURCE-$0}"
this_dir=$(cd -P -- "$(dirname -- "${this}")" && pwd -P)

names=(
    cifar_conv
    cifar_conv_da
    cifar_resnet
    cifar_resnet50_da_bn
    cifar_resnet_da_bn
    cifar_vgg16
    cifar_vgg16_bn
    cifar_vgg16_gn
)

for n in "${names[@]}"; do
    echo -en "\n*** ${n} ***\n"
    python3 "${this_dir}"/${n}.py --gpu --epochs 1
done
