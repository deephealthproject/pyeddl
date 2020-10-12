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
Autoencoder siamese example for MNIST.
"""

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor


def main(args):
    eddl.download_mnist()

    # encoder
    in_ = eddl.Input([784])
    layer = eddl.Activation(eddl.Dense(in_, 256), "relu")
    layer = eddl.Activation(eddl.Dense(layer, 128), "relu")
    enc = eddl.Model([in_], [layer])

    # decoder
    in_ = eddl.Input([128])
    out = eddl.Activation(eddl.Dense(in_, 64), "relu")
    dec = eddl.Model([in_], [out])

    # merge
    base = eddl.Model([enc, dec])
    eddl.setName(base, "base")

    in1 = eddl.Input([784])
    in2 = eddl.Input([784])

    # get siameses
    out1 = eddl.getLayer(base, [in1])
    out2 = eddl.getLayer(base, [in2])

    # combine siameses with some operations
    layer = eddl.Diff(out1, out2)
    layer = eddl.ReLu(eddl.Dense(layer, 256))
    outs = eddl.Sigmoid(eddl.Dense(layer, 784))

    # build final model
    siamese = eddl.Model([in1, in2], [outs])
    eddl.setName(siamese, "siamese")

    eddl.build(
        siamese,
        eddl.adam(0.0001),
        ["dice"],
        ["dice"],
        eddl.CS_GPU([1]) if args.gpu else eddl.CS_CPU()
    )
    eddl.summary(siamese)
    eddl.plot(siamese, "model.pdf")

    x_train = Tensor.load("mnist_trX.bin")
    if args.small:
        x_train = x_train.select([":6000"])
    x_train.div_(255.0)
    # eddl.fit(
    #     siamese, [x_train, x_train], [x_train], args.batch_size, args.epochs
    # )
    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--small", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
