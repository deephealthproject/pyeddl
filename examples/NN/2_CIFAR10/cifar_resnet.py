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
ResNet18 for CIFAR10.
"""

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor


def ResBlock(layer, filters, nconv, half):
    in_ = layer
    strides = [2, 2] if half else [1, 1]
    layer = eddl.ReLu(eddl.Conv(layer, filters, [3, 3], strides))
    for i in range(nconv - 1):
        layer = eddl.ReLu(eddl.Conv(layer, filters, [3, 3], [1, 1]))
    if (half):
        return eddl.Add(eddl.Conv(in_, filters, [1, 1], [2, 2]), layer)
    else:
        return eddl.Add(layer, in_)


MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")


def main(args):
    eddl.download_cifar10()

    num_classes = 10

    in_ = eddl.Input([3, 32, 32])

    layer = in_
    layer = eddl.ReLu(eddl.Conv(layer, 64, [3, 3], [1, 1]))
    layer = eddl.Pad(layer, [0, 1, 1, 0])
    layer = ResBlock(layer, 64, 2, True)
    layer = ResBlock(layer, 64, 2, False)
    layer = ResBlock(layer, 128, 2, True)
    layer = ResBlock(layer, 128, 2, False)
    layer = ResBlock(layer, 256, 2, True)
    layer = ResBlock(layer, 256, 2, False)
    layer = ResBlock(layer, 512, 2, True)
    layer = ResBlock(layer, 512, 2, False)
    layer = eddl.Reshape(layer, [-1])
    layer = eddl.Activation(eddl.Dense(layer, 512), "relu")

    out = eddl.Softmax(eddl.Dense(layer, num_classes))
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )

    eddl.summary(net)
    eddl.plot(net, "model.pdf")

    x_train = Tensor.load("cifar_trX.bin")
    y_train = Tensor.load("cifar_trY.bin")
    x_train.div_(255.0)

    x_test = Tensor.load("cifar_tsX.bin")
    y_test = Tensor.load("cifar_tsY.bin")
    x_test.div_(255.0)

    if args.small:
        x_train = x_train.select([":5000"])
        y_train = y_train.select([":5000"])
        x_test = x_test.select([":1000"])
        y_test = y_test.select([":1000"])

    for i in range(args.epochs):
        eddl.fit(net, [x_train], [y_train], args.batch_size, 1)
        eddl.evaluate(net, [x_test], [y_test], bs=args.batch_size)
    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=100)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES),
                        choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
