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
AE example with Python loss and metric.
"""

import argparse
import sys

import pyeddl.eddl as eddl
import pyeddl.eddlT as eddlT
from pyeddl._core import Loss, Metric


class MSELoss(Loss):

    def __init__(self):
        Loss.__init__(self, "py_mean_squared_error")

    def delta(self, t, y, d):
        eddlT.copyTensor(eddlT.sub(y, t), d)
        eddlT.div_(d, eddlT.getShape(t)[0])

    def value(self, t, y):
        aux = eddlT.add(t, eddlT.neg(y))
        aux = eddlT.mult(aux, aux)
        return aux.sum() / eddlT.getShape(t)[0]


class MSEMetric(Metric):

    def __init__(self):
        Metric.__init__(self, "py_mean_squared_error")

    def value(self, t, y):
        aux = eddlT.add(t, eddlT.neg(y))
        aux = eddlT.mult(aux, aux)
        return aux.sum() / eddlT.getShape(t)[0]


def main(args):
    eddl.download_mnist()

    in_ = eddl.Input([784])

    layer = in_
    layer = eddl.Activation(eddl.Dense(layer, 256), "relu")
    layer = eddl.Activation(eddl.Dense(layer, 128), "relu")
    layer = eddl.Activation(eddl.Dense(layer, 64), "relu")
    layer = eddl.Activation(eddl.Dense(layer, 128), "relu")
    layer = eddl.Activation(eddl.Dense(layer, 256), "relu")
    out = eddl.Dense(layer, 784)

    net = eddl.Model([in_], [out])
    mse_loss = MSELoss()
    mse_metric = MSEMetric()
    net.build(
        eddl.sgd(0.001, 0.9),
        [mse_loss],
        [mse_metric],
        eddl.CS_GPU([1]) if args.gpu else eddl.CS_CPU()
    )

    eddl.summary(net)
    eddl.plot(net, "model.pdf")

    x_train = eddlT.load("trX.bin")
    eddlT.div_(x_train, 255.0)
    eddl.fit(net, [x_train], [x_train], args.batch_size, args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
