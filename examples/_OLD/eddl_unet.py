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
UNET example.
"""

import argparse
import sys

import numpy as np
from pyeddl.api import (
    Input, Activation, Conv, MaxPool, Dropout, UpSampling, Concat, Model,
    CS_CPU, CS_GPU, sgd, build, DEV_CPU, T, fit
)
from pyeddl._core import Tensor


def main(args):

    dimX, dimY = 416, 416
    filters = 32
    ks = [3, 3]
    n_classes = 1

    in_ = Input([3, dimX, dimY])
    l1 = in_
    l2 = MaxPool(Activation(Conv(l1, filters * 2, ks), "relu"), [2, 2])
    l3 = MaxPool(Activation(Conv(l2, filters * 4, ks), "relu"), [2, 2])
    l4 = MaxPool(Activation(Conv(l3, filters * 8, ks), "relu"), [2, 2])
    l5 = MaxPool(Activation(Conv(l4, filters * 16, ks), "relu"), [2, 2])
    l6 = Dropout(l5, 0.5)
    l7 = UpSampling(l6, [2, 2])
    l8 = Concat([l4, l7])
    l9 = Activation(Conv(l8, filters * 8, ks), "relu")
    l10 = UpSampling(l9, [2, 2])
    l11 = Concat([l3, l10])
    l12 = Activation(Conv(l11, filters * 4, ks), "relu")
    l13 = UpSampling(l12, [2, 2])
    l14 = Concat([l2, l13])
    l15 = Activation(Conv(l14, filters * 2, ks), "relu")
    l16 = UpSampling(l15, [2, 2])
    l17 = Concat([l1, l16])
    l18 = Activation(Conv(l17, filters, ks), "relu")
    # next layer should use the "sigmoid" activation function
    l19 = Activation(Conv(l18, n_classes, [1, 1]), "softmax")
    out = l19
    net = Model([in_], [out])

    build(
        net,
        sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        CS_GPU([1]) if args.gpu else CS_CPU()
    )

    print(net.summary())

    print("reading dataset")
    tr_fn = "training_set.npz"
    data = np.load(tr_fn)
    trX_a = data["img"]  # [n_samples, n_channels, width, height]
    trY_a = data["msk"]  # [n_samples, n_classes, width, height]
    trX_t = Tensor(trX_a.astype(np.float32), DEV_CPU)
    trY_t = Tensor(trY_a.astype(np.float32), DEV_CPU)
    trX = T(list(trX_a.shape))
    trX.input = trX_t
    trY = T(list(trY_a.shape))
    trY.input = trY_t
    fit(net, [trX], [trY], args.batch_size, args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=16)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
