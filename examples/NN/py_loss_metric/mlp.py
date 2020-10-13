# Copyright (c) 2019-2020 CRS4
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
MLP example with Python metric.
"""

import argparse
import sys

import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from pyeddl._core import Metric


class CategoricalAccuracy(Metric):

    def __init__(self):
        Metric.__init__(self, "py_categorical_accuracy")

    def value(self, t, y):
        a = t.getdata()
        b = y.getdata()
        return (np.argmax(a, axis=-1) == np.argmax(b, axis=-1)).sum()


def main(args):
    eddl.download_mnist()

    num_classes = 10

    in_ = eddl.Input([784])

    layer = in_
    layer = eddl.BatchNormalization(
        eddl.Activation(eddl.L2(eddl.Dense(layer, 1024), 0.0001), "relu"), True
    )
    layer = eddl.BatchNormalization(
        eddl.Activation(eddl.L2(eddl.Dense(layer, 1024), 0.0001), "relu"), True
    )
    layer = eddl.BatchNormalization(
        eddl.Activation(eddl.L2(eddl.Dense(layer, 1024), 0.0001), "relu"), True
    )
    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")
    net = eddl.Model([in_], [out])

    acc = CategoricalAccuracy()
    net.build(
        eddl.sgd(0.01, 0.9),
        [eddl.getLoss("soft_cross_entropy")],
        [acc],
        eddl.CS_GPU([1]) if args.gpu else eddl.CS_CPU()
    )

    eddl.summary(net)
    eddl.plot(net, "model.pdf")

    x_train = Tensor.load("mnist_trX.bin")
    y_train = Tensor.load("mnist_trY.bin")
    x_test = Tensor.load("mnist_tsX.bin")
    y_test = Tensor.load("mnist_tsY.bin")

    x_train.div_(255.0)
    x_test.div_(255.0)

    eddl.fit(net, [x_train], [y_train], args.batch_size, args.epochs)

    eddl.evaluate(net, [x_test], [y_test])
    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
