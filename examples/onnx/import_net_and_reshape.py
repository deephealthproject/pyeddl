# Copyright (c) 2019-2021 CRS4
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
Import resnet18 model from file and reshape its input to train on cifar10.
"""

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor, DEV_CPU


MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")


def main(args):

    freeze_epochs = 2
    unfreeze_epochs = 5
    num_classes = 10  # 10 labels in cifar10

    eddl.download_cifar10()
    eddl.download_model("resnet18.onnx", "re7jodd12srksd7")
    net = eddl.import_net_from_onnx_file(
        "resnet18.onnx", [3, 32, 32], DEV_CPU
    )
    names = [_.name for _ in net.layers]

    # Remove dense output layer
    eddl.removeLayer(net, "resnetv15_dense0_fwd")
    # Get last layer to connect the new dense
    layer = eddl.getLayer(net, "flatten_170")
    out = eddl.Softmax(eddl.Dense(layer, num_classes, True, "new_dense"))
    # Get input layer
    in_ = eddl.getLayer(net, "data")
    # Create a new model
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.adam(0.0001),
        ["softmax_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem),
        False  # do not initialize weights to random values
    )
    eddl.summary(net)
    # Force initialization of new layers
    eddl.initializeLayer(net, "new_dense")

    x_train = Tensor.load("cifar_trX.bin")
    y_train = Tensor.load("cifar_trY.bin")
    x_test = Tensor.load("cifar_tsX.bin")
    y_test = Tensor.load("cifar_tsY.bin")
    if args.small:
        sel = [f":{2 * args.batch_size}"]
        x_train = x_train.select(sel)
        y_train = y_train.select(sel)
        x_test = x_test.select(sel)
        y_test = y_test.select(sel)

    x_train.div_(255.0)
    x_test.div_(255.0)

    # Freeze pretrained weights
    for n in names:
        eddl.setTrainable(net, n, False)

    # Train new layers
    eddl.fit(net, [x_train], [y_train], args.batch_size, freeze_epochs)

    # Unfreeze weights
    for n in names:
        eddl.setTrainable(net, n, True)

    # Train all layers
    eddl.fit(net, [x_train], [y_train], args.batch_size, unfreeze_epochs)

    # Evaluate
    eddl.evaluate(net, [x_test], [y_test], args.batch_size)

    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=100)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--input", metavar="STRING",
                        default="trained_model.onnx",
                        help="input path of the serialized model")
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES),
                        choices=MEM_CHOICES, default="low_mem")
    parser.add_argument("--small", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
