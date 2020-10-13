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
Embedding + RNN example.
"""

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor


def main(args):
    eddl.download_imdb_2000()

    length = 250
    embdim = 32
    vocsize = 2000

    in_ = eddl.Input([1])  # 1 word
    layer = in_
    layer = eddl.RandomUniform(
        eddl.Embedding(layer, vocsize, 1, embdim), -0.05, 0.05
    )
    layer = eddl.RNN(layer, 32)
    layer = eddl.ReLu(eddl.Dense(layer, 256))
    out = eddl.Sigmoid(eddl.Dense(layer, 1))
    net = eddl.Model([in_], [out])
    eddl.build(
        net,
        eddl.adam(0.001),
        ["cross_entropy"],
        ["binary_accuracy"],
        eddl.CS_GPU() if args.gpu else eddl.CS_CPU()
    )
    eddl.summary(net)

    x_train = Tensor.load("imdb_2000_trX.bin")
    y_train = Tensor.load("imdb_2000_trY.bin")
    x_test = Tensor.load("imdb_2000_tsX.bin")
    y_test = Tensor.load("imdb_2000_tsY.bin")
    if args.small:
        x_train = x_train.select([":500"])
        y_train = y_train.select([":500"])
        x_test = x_test.select([":200"])
        y_test = y_test.select([":200"])

    #  batch x timesteps x input_dim
    x_train.reshape_([x_train.shape[0], length, 1])
    x_test.reshape_([x_test.shape[0], length, 1])

    for i in range(args.epochs):
        eddl.fit(net, [x_train], [y_train], args.batch_size, 1)
        eddl.evaluate(net, [x_test], [y_test])
    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--small", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
