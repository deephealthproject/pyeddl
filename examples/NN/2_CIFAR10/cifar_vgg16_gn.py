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
VGG16 for CIFAR10 with group normalization.
"""

import argparse
import sys

import pyeddl.eddl as eddl
import pyeddl.eddlT as eddlT


def Block1(layer, filters):
    return eddl.ReLu(
        eddl.GroupNormalization(eddl.Conv(layer, filters, [1, 1], [1, 1]), 4)
    )


def Block3_2(layer, filters):
    layer = eddl.ReLu(
        eddl.GroupNormalization(eddl.Conv(layer, filters, [3, 3], [1, 1]), 4)
    )
    layer = eddl.ReLu(
        eddl.GroupNormalization(eddl.Conv(layer, filters, [3, 3], [1, 1]), 4)
    )
    return layer


def main(args):
    eddl.download_cifar10()

    num_classes = 10

    in_ = eddl.Input([3, 32, 32])

    layer = in_
    layer = eddl.RandomCropScale(layer, [0.8, 1.0])
    layer = eddl.RandomFlip(layer, 1)
    layer = eddl.MaxPool(Block3_2(layer, 64))
    layer = eddl.MaxPool(Block3_2(layer, 128))
    layer = eddl.MaxPool(Block1(Block3_2(layer, 256), 256))
    layer = eddl.MaxPool(Block1(Block3_2(layer, 512), 512))
    layer = eddl.MaxPool(Block1(Block3_2(layer, 512), 512))
    layer = eddl.Reshape(layer, [-1])
    layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer, 512)))

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
    eddl.plot(net, "model.pdf", "TB")

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
    # batch size should be small to test group normalization
    parser.add_argument("--batch-size", type=int, metavar="INT", default=8)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
