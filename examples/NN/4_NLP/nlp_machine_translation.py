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
Machine translation example.
"""

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor


MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")


def main(args):
    eddl.download_eutrans()

    epochs = 1 if args.small else 5

    ilength = 30
    olength = 30
    invs = 687
    outvs = 514
    embedding = 64

    # Encoder
    in_ = eddl.Input([1])  # 1 word
    layer = in_
    lE = eddl.RandomUniform(
        eddl.Embedding(layer, invs, 1, embedding, True), -0.05, 0.05
    )
    enc = eddl.LSTM(lE, 128, True)
    cps = eddl.GetStates(enc)

    # Decoder
    ldin = eddl.Input([outvs])
    ld = eddl.ReduceArgMax(ldin, [0])
    ld = eddl.RandomUniform(
        eddl.Embedding(ld, outvs, 1, embedding), -0.05, 0.05
    )
    layer = eddl.LSTM([ld, cps], 128)
    out = eddl.Softmax(eddl.Dense(layer, outvs))
    eddl.setDecoder(ldin)

    net = eddl.Model([in_], [out])

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
    x_train = Tensor.load("eutrans_trX.bin")
    y_train = Tensor.load("eutrans_trY.bin")
    y_train = Tensor.onehot(y_train, outvs)
    # batch x timesteps x input_dim
    x_train.reshape_([x_train.shape[0], ilength, 1])
    # batch x timesteps x ouput_dim
    y_train.reshape_([y_train.shape[0], olength, outvs])

    x_test = Tensor.load("eutrans_tsX.bin")
    y_test = Tensor.load("eutrans_tsY.bin")
    y_test = Tensor.onehot(y_test, outvs)
    # batch x timesteps x input_dim
    x_test.reshape_([x_test.shape[0], ilength, 1])
    # batch x timesteps x ouput_dim
    y_test.reshape_([y_test.shape[0], olength, outvs])

    if args.small:
        sel = [f":{3 * args.batch_size}", ":", ":"]
        x_train = x_train.select(sel)
        y_train = y_train.select(sel)
        x_test = x_test.select(sel)
        y_test = y_test.select(sel)

    # Train model
    ybatch = Tensor([args.batch_size, olength, outvs])
    eddl.next_batch([y_train], [ybatch])
    for i in range(epochs):
        eddl.fit(net, [x_train], [y_train], args.batch_size, 1)

    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--small", action="store_true")
    # This needs full_mem, otherwise it crashes with a weird "Tensors with
    # different size (Tensor::copy)" error during fit
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES),
                        choices=MEM_CHOICES, default="full_mem")
    main(parser.parse_args(sys.argv[1:]))
