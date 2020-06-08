# Copyright (c) 2019-2020, CRS4
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""\
ResNet18 for CIFAR10.
"""

import argparse
import sys

import pyeddl.eddl as eddl
import pyeddl.eddlT as eddlT


def ResBlock(layer, filters, nconv, half):
    in_ = layer
    strides = [2, 2] if half else [1, 1]
    layer = eddl.ReLu(eddl.Conv(layer, filters, [3, 3], strides))
    for i in range(nconv - 1):
        layer = eddl.ReLu(eddl.Conv(layer, filters, [3, 3], [1, 1]))
    if (half):
        return eddl.Sum(eddl.Conv(in_, filters, [1, 1], [2, 2]), layer)
    else:
        return eddl.Sum(layer, in_)


def main(args):
    eddl.download_cifar10()

    num_classes = 10

    in_ = eddl.Input([3, 32, 32])

    layer = in_
    layer = eddl.ReLu(eddl.Conv(layer, 64, [3, 3], [1, 1]))
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

    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU() if args.gpu else eddl.CS_CPU()
    )

    eddl.summary(net)
    eddl.plot(net, "model.pdf")

    x_train = eddlT.load("cifar_trX.bin")
    y_train = eddlT.load("cifar_trY.bin")
    eddlT.div_(x_train, 255.0)

    x_test = eddlT.load("cifar_tsX.bin")
    y_test = eddlT.load("cifar_tsY.bin")
    eddlT.div_(x_test, 255.0)

    for i in range(args.epochs):
        eddl.fit(net, [x_train], [y_train], args.batch_size, 1)
        eddl.evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=100)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
