#!/usr/bin/env bash

# Assumes EDDL compiled for GPU.

set -euo pipefail
[ -n "${DEBUG:-}" ] && set -x
this="${BASH_SOURCE-$0}"
this_dir=$(cd -P -- "$(dirname -- "${this}")" && pwd -P)

img="${this_dir}/../../third_party/eddl/examples/data/elephant.jpg"
classes="${this_dir}/../../third_party/eddl/examples/data/imagenet_class_names.txt"
model="resnet34-v1-7.onnx"
[ -f "${model}" ] || wget -q "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/${model}"
python3 image_classification_resnet34.py --gpu "${img}" "${classes}" "${model}"
