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
Basic MLP for MNIST with data augmentation.
"""

import argparse
import sys

import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT


def main(args):
    eddl.download_mnist()

    num_classes = 10

    in_ = eddl.Input([784])

    layer = in_
    layer = eddl.Reshape(layer, [1, 28, 28])
    layer = eddl.RandomCropScale(layer, [0.9, 1.0])
    layer = eddl.Reshape(layer, [-1])
    layer = eddl.ReLu(eddl.GaussianNoise(
        eddl.BatchNormalization(eddl.Dense(layer, 1024)), 0.3
    ))
    layer = eddl.ReLu(eddl.GaussianNoise(
        eddl.BatchNormalization(eddl.Dense(layer, 1024)), 0.3
    ))
    layer = eddl.ReLu(eddl.GaussianNoise(
        eddl.BatchNormalization(eddl.Dense(layer, 1024)), 0.3
    ))
    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU([1], "low_mem") if args.gpu else eddl.CS_CPU(-1, "low_mem")
    )

    eddl.summary(net)
    eddl.plot(net, "model.pdf")

    x_train = eddlT.load("trX.bin")
    y_train = eddlT.load("trY.bin")
    x_test = eddlT.load("tsX.bin")
    y_test = eddlT.load("tsY.bin")

    eddlT.div_(x_train, 255.0)
    eddlT.div_(x_test, 255.0)

    eddl.fit(net, [x_train], [y_train], args.batch_size, args.epochs)
    eddl.evaluate(net, [x_test], [y_test])

    # LR annealing
    if args.epochs < 4:
        return
    eddl.setlr(net, [0.005, 0.9])
    eddl.fit(net, [x_train], [y_train], args.batch_size, args.epochs // 2)
    eddl.evaluate(net, [x_test], [y_test])
    eddl.setlr(net, [0.001, 0.9])
    eddl.fit(net, [x_train], [y_train], args.batch_size, args.epochs // 2)
    eddl.evaluate(net, [x_test], [y_test])
    eddl.setlr(net, [0.0001, 0.9])
    eddl.fit(net, [x_train], [y_train], args.batch_size, args.epochs // 4)
    eddl.evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
