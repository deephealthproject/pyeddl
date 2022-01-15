# Copyright (c) 2019-2022 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""\
Application example.
"""

import argparse
import sys

import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor


def preprocess_input_resnet34(input_, target_size):
    mean_vec = Tensor.fromarray(
        np.array([0.485, 0.456, 0.406], dtype=np.float32),
        input_.device
    )
    std_vec = Tensor.fromarray(
        np.array([0.229, 0.224, 0.225], dtype=np.float32),
        input_.device
    )
    if input_.ndim not in {3, 4}:
        raise RuntimeError("Input tensor must be 3D or 4D")
    if input_.ndim == 3:
        input_.unsqueeze_(0)  # convert to 4D
    new_input = input_.scale(target_size)  # (height, width)
    # Normalization [0..1]
    new_input.mult_(1 / 255.0)
    # Standardization: (X - mean) / std
    mean = Tensor.broadcast(mean_vec, new_input)
    std = Tensor.broadcast(std_vec, new_input)
    new_input.sub_(mean)
    new_input.div_(std)
    return new_input


def main(args):
    in_channels = 3
    in_height = 224
    in_width = 224
    print("Importing ONNX model")
    net = eddl.import_net_from_onnx_file(
        args.model_fn,
        [in_channels, in_height, in_width]
    )
    # Add a softmax layer to get probabilities directly from the model
    input_ = net.lin[0]   # getLayer(net,"input_layer_name")
    output = net.lout[0]   # getLayer(net,"output_layer_name")
    new_output = eddl.Softmax(output)

    net = eddl.Model([input_], [new_output])
    eddl.build(
        net,
        eddl.adam(0.001),  # not used for prediction
        ["softmax_cross_entropy"],  # not used for prediction
        ["categorical_accuracy"],  # not used for prediction
        eddl.CS_GPU() if args.gpu else eddl.CS_CPU(),
        False  # Disable model initialization, we want to use the ONNX weights
    )
    eddl.summary(net)

    image = Tensor.load(args.img_fn)
    image_preprocessed = preprocess_input_resnet34(
        image,
        [in_height, in_width]
    )
    outputs = eddl.predict(net, [image_preprocessed])
    print("Reading class names...")
    with open(args.classes_fn, "rt") as f:
        class_names = [_.strip() for _ in f]
    print("Top 5 predictions:")
    print(eddl.get_topk_predictions(outputs[0], class_names, 5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("img_fn", metavar="IMAGE", help="image to classify")
    parser.add_argument("classes_fn", metavar="CLASS_NAMES",
                        help="text file containing ImageNet class names")
    parser.add_argument("model_fn", metavar="MODEL", help="ONNX model file")
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
