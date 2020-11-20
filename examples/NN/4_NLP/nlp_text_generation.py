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
Text generation.
"""

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor


MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")


def ResBlock(layer, filters, nconv, half):
    in_ = layer

    if half:
        layer = eddl.ReLu(eddl.BatchNormalization(
            eddl.Conv(layer, filters, [3, 3], [2, 2]), True
        ))
    else:
        layer = eddl.ReLu(eddl.BatchNormalization(
            eddl.Conv(layer, filters, [3, 3], [1, 1]), True
        ))

    for i in range(nconv - 1):
        layer = eddl.ReLu(eddl.BatchNormalization(
            eddl.Conv(layer, filters, [3, 3], [1, 1]), True
        ))

    if half:
        return eddl.Add(eddl.BatchNormalization(
            eddl.Conv(in_, filters, [1, 1], [2, 2]), True
        ), layer)
    else:
        return eddl.Add(layer, in_)


def main(args):
    eddl.download_flickr()

    olength = 20
    outvs = 2000
    embdim = 32

    # Define network
    in_ = eddl.Input([3, 256, 256])  # Image
    layer = in_
    layer = eddl.ReLu(eddl.Conv(layer, 64, [3, 3], [2, 2]))

    layer = ResBlock(layer, 64, 2, 1)
    layer = ResBlock(layer, 64, 2, 0)
    layer = ResBlock(layer, 128, 2, 1)
    layer = ResBlock(layer, 128, 2, 0)
    layer = ResBlock(layer, 256, 2, 1)
    layer = ResBlock(layer, 256, 2, 0)
    layer = ResBlock(layer, 512, 2, 1)
    layer = ResBlock(layer, 512, 2, 0)
    layer = eddl.GlobalAveragePool(layer)
    layer = eddl.Reshape(layer, [-1])

    # Decoder
    ldec = eddl.Input([outvs])
    ldec = eddl.ReduceArgMax(ldec, [0])
    ldec = eddl.RandomUniform(
        eddl.Embedding(ldec, outvs, 1, embdim), -0.05, 0.05
    )
    layer = eddl.Decoder(eddl.LSTM(ldec, 512, True), layer, "concat")
    out = eddl.Softmax(eddl.Dense(layer, outvs))
    net = eddl.Model([in_], [out])

    # Build model
    eddl.build(
        net,
        eddl.adam(0.001),
        ["softmax_cross_entropy"],
        ["accuracy"],
        eddl.CS_GPU(mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )
    eddl.summary(net)

    # Load dataset
    x_train = Tensor.load("flickr_trX.bin")
    y_train = Tensor.load("flickr_trY.bin")
    if args.small:
        x_train = x_train.select([":200"])
        y_train = y_train.select([":200"])
    xtrain = Tensor.permute(x_train, [0, 3, 1, 2])
    y_train = Tensor.onehot(y_train, outvs)
    # batch x timesteps x input_dim
    y_train.reshape_([y_train.shape[0], olength, outvs])

    # Train model
    for i in range(args.epochs):
        eddl.fit(net, [xtrain], [y_train], args.batch_size, 1)

    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=24)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES),
                        choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
