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


def main(args):
    eddl.download_flickr()

    epochs = 2 if args.small else 50

    olength = 20
    outvs = 2000
    embdim = 32

    # True: remove last layers and set new top = flatten
    # new input_size: [3, 256, 256] (from [224, 224, 3])
    net = eddl.download_resnet18(True, [3, 256, 256])
    lreshape = eddl.getLayer(net, "top")

    # create a new model from input output
    image_in = eddl.getLayer(net, "input")

    # Decoder
    ldecin = eddl.Input([outvs])
    ldec = eddl.ReduceArgMax(ldecin, [0])
    ldec = eddl.RandomUniform(
        eddl.Embedding(ldec, outvs, 1, embdim), -0.05, 0.05
    )

    # layer = eddl.Decoder(eddl.LSTM(ldec, 512, True), layer, "concat")
    ldec = eddl.Concat([ldec, lreshape])
    layer = eddl.LSTM(ldec, 512, True)
    out = eddl.Softmax(eddl.Dense(layer, outvs))
    eddl.setDecoder(ldecin)
    net = eddl.Model([image_in], [out])

    # Build model
    eddl.build(
        net,
        eddl.adam(0.01),
        ["softmax_cross_entropy"],
        ["accuracy"],
        eddl.CS_GPU(mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )
    eddl.summary(net)

    # Load dataset
    x_train = Tensor.load("flickr_trX.bin")
    y_train = Tensor.load("flickr_trY.bin")
    if args.small:
        x_train = x_train.select([f":{2 * args.batch_size}", ":", ":", ":"])
        y_train = y_train.select([f":{2 * args.batch_size}"])
    xtrain = Tensor.permute(x_train, [0, 3, 1, 2])
    y_train = Tensor.onehot(y_train, outvs)
    # batch x timesteps x input_dim
    y_train.reshape_([y_train.shape[0], olength, outvs])

    eddl.fit(net, [xtrain], [y_train], args.batch_size, epochs)
    # eddl.save(net, "img2text.bin", "bin")

    # print("\n === INFERENCE ===\n")

    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=24)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES),
                        choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
