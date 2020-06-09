# Copyright (c) 2019-2020, CRS4
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
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
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
