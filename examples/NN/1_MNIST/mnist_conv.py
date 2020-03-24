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
MNIST CONV example.
"""

import argparse
import sys

import pyeddl.eddl as eddl
import pyeddl.eddlT as eddlT


def main(args):
    eddl.download_mnist()

    num_classes = 10

    in_ = eddl.Input([784])
    layer = in_
    layer = eddl.Reshape(layer, [1, 28, 28])
    layer = eddl.MaxPool(
        eddl.ReLu(eddl.Conv(layer, 32, [3, 3], [1, 1])),
        [3, 3], [1, 1], "same"
    )
    layer = eddl.MaxPool(
        eddl.ReLu(eddl.Conv(layer, 64, [3, 3], [1, 1])),
        [2, 2], [2, 2], "same"
    )
    layer = eddl.MaxPool(
        eddl.ReLu(eddl.Conv(layer, 128, [3, 3], [1, 1])),
        [3, 3], [2, 2], "none"
    )
    layer = eddl.MaxPool(
        eddl.ReLu(eddl.Conv(layer, 256, [3, 3], [1, 1])),
        [2, 2], [2, 2], "none"
    )
    layer = eddl.Reshape(layer, [-1])
    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.rmsprop(0.01),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU() if args.gpu else eddl.CS_CPU()
    )

    eddl.summary(net)

    x_train = eddlT.load("trX.bin")
    y_train = eddlT.load("trY.bin")
    x_test = eddlT.load("tsX.bin")
    y_test = eddlT.load("tsY.bin")
    eddlT.div_(x_train, 255.0)
    eddlT.div_(x_test, 255.0)

    eddl.fit(net, [x_train], [y_train], args.batch_size, args.epochs)
    eddl.evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
