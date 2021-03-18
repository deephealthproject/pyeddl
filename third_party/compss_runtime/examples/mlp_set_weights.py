# Copyright (c) 2020 CRS4
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
Import model from file in ONNX format.
"""

import argparse
import os
import sys

import pyeddl._core.eddl as eddl
from pyeddl.tensor import Tensor as eddlT

from net_utils import net_parametersToNumpy
from net_utils import net_parametersToTensor


def main(args):
    if not os.path.isfile("mlp_transfer_weights.onnx"):
        raise RuntimeError("input file '%s' not found" % "mlp_transfer_weights.onnx")

    eddl.download_mnist()

    print("importing net from file")
    net = eddl.import_net_from_onnx_file("mlp_transfer_weights.onnx")
    print("output size =", len(net.lout))

    eddl.build(
        net,
        eddl.sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU(),
        False  # do not initialize weights to random values
    )

    #net.resize(args.batch_size)  # resize manually since we don't use "fit"
    eddl.summary(net)

    x_test = eddlT.load("tsX.bin")
    y_test = eddlT.load("tsY.bin")

    eddlT.div_(x_test, 255.0)

    eddl.evaluate(net, [x_test], [y_test])

    print("Checking setting weights to a new network")
    
    num_classes = 10

    in_ = eddl.Input([784])
    layer = in_
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")
    net2 = eddl.Model([in_], [out])

    eddl.build(
        net2,
        eddl.sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU()
    )

    v = net_parametersToTensor(net_parametersToNumpy(net.getParameters()))
    net2.setParameters(v)

    #net2.resize(args.batch_size)  # resize manually since we don't use "fit"
    eddl.summary(net2)

    x_test = eddlT.load("tsX.bin")
    y_test = eddlT.load("tsY.bin")

    eddlT.div_(x_test, 255.0)

    eddl.evaluate(net2, [x_test], [y_test])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--input", metavar="STRING",
                        default="trained_model.onnx",
                        help="input path of the serialized model")
    main(parser.parse_args(sys.argv[1:]))
