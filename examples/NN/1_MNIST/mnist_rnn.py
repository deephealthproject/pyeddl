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
MNIST RNN example.
"""

import argparse
import sys

import pyeddl.eddl as eddl
import pyeddl.eddlT as eddlT
from pyeddl._core import Tensor


def main(args):
    eddl.download_mnist()

    num_classes = 10

    in_ = eddl.Input([28])

    layer = in_
    layer = eddl.LeakyReLu(eddl.Dense(layer, 32))
    layer = eddl.LeakyReLu(eddl.RNN(layer, 32))
    layer = eddl.LeakyReLu(eddl.Dense(layer, 32))
    out = eddl.Softmax(eddl.Dense(layer, num_classes))
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.rmsprop(0.001),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU() if args.gpu else eddl.CS_CPU()
    )

    eddl.summary(net)
    eddl.plot(net, "model.pdf")

    x_train = eddlT.load("trX.bin")
    y_train = eddlT.load("trY.bin")
    x_test = eddlT.load("tsX.bin")
    y_test = eddlT.load("tsY.bin")

    eddlT.reshape_(x_train, [60000, 28, 28])
    x_trainp = Tensor.permute(x_train, [1, 0, 2])

    eddlT.reshape_(x_test, [10000, 28, 28])
    x_testp = Tensor.permute(x_test, [1, 0, 2])

    eddlT.div_(x_trainp, 255.0)
    eddlT.div_(x_testp, 255.0)

    for i in range(args.epochs):
        eddl.fit(net, [x_trainp], [y_train], args.batch_size, 1)
        eddl.evaluate(net, [x_testp], [y_test])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
