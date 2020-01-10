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
AE example with Python metric.
"""

import argparse
import sys

from pyeddl.api import (
    Input, Activation, Dense, Model, sgd, CS_CPU, T_load, div, fit, CS_GPU
)
from pyeddl.utils import download_mnist, loss_func
from pyeddl._core import CustomMetric, Tensor


def py_mse(t, y):
    aux = Tensor(t.getShape(), t.device)
    Tensor.add(1, t, -1, y, aux, 0)
    Tensor.el_mult(aux, aux, aux, 0)
    return aux.sum()


def main(args):
    download_mnist()

    epochs = args.epochs
    batch_size = args.batch_size

    in_ = Input([784])
    layer = in_
    layer = Activation(Dense(layer, 256), "relu")
    layer = Activation(Dense(layer, 128), "relu")
    layer = Activation(Dense(layer, 64), "relu")
    layer = Activation(Dense(layer, 128), "relu")
    layer = Activation(Dense(layer, 256), "relu")
    out = Dense(layer, 784)
    net = Model([in_], [out])

    mse = CustomMetric(py_mse, "py_mean_squared_error")

    net.build(
        sgd(0.001, 0.9),
        [loss_func("mean_squared_error")],
        [mse],
        CS_GPU([1]) if args.gpu else CS_CPU()
    )

    print(net.summary())

    x_train = T_load("trX.bin")
    div(x_train, 255.0)
    fit(net, [x_train], [x_train], batch_size, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=100)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
