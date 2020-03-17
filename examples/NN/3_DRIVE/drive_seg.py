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
Drive segmentation.

https://drive.grand-challenge.org/DRIVE
"""

import argparse
import sys

import pyeddl.eddl as eddl
import pyeddl.eddlT as eddlT
from pyeddl._core import Tensor


USE_CONCAT = 1


def UNetWithPadding(layer):
    x = layer
    depth = 32

    x = eddl.LeakyReLu(eddl.Conv(x, depth, [3, 3], [1, 1], "same"))
    x = eddl.LeakyReLu(eddl.Conv(x, depth, [3, 3], [1, 1], "same"))
    x2 = eddl.MaxPool(x, [2, 2], [2, 2])
    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"))
    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"))
    x3 = eddl.MaxPool(x2, [2, 2], [2, 2])
    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"))
    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"))
    x4 = eddl.MaxPool(x3, [2, 2], [2, 2])
    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"))
    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"))
    x5 = eddl.MaxPool(x4, [2, 2], [2, 2])
    x5 = eddl.LeakyReLu(eddl.Conv(x5, 8*depth, [3, 3], [1, 1], "same"))
    x5 = eddl.LeakyReLu(eddl.Conv(x5, 8*depth, [3, 3], [1, 1], "same"))
    x5 = eddl.Conv(
        eddl.UpSampling(x5, [2, 2]), 8*depth, [2, 2], [1, 1], "same"
    )

    x4 = eddl.Concat([x4, x5]) if USE_CONCAT else eddl.Sum([x4, x5])
    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"))
    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"))
    x4 = eddl.Conv(
        eddl.UpSampling(x4, [2, 2]), 4*depth, [2, 2], [1, 1], "same"
    )

    x3 = eddl.Concat([x3, x4]) if USE_CONCAT else eddl.Sum([x3, x4])
    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"))
    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"))
    x3 = eddl.Conv(
        eddl.UpSampling(x3, [2, 2]), 2*depth, [2, 2], [1, 1], "same"
    )

    x2 = eddl.Concat([x2, x3]) if USE_CONCAT else eddl.Sum([x2, x3])
    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"))
    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"))
    x2 = eddl.Conv(
        eddl.UpSampling(x2, [2, 2]), depth, [2, 2], [1, 1], "same"
    )

    x = eddl.Concat([x, x2]) if USE_CONCAT else eddl.Sum([x, x2])
    x = eddl.LeakyReLu(eddl.Conv(x, depth, [3, 3], [1, 1], "same"))
    x = eddl.LeakyReLu(eddl.Conv(x, depth, [3, 3], [1, 1], "same"))
    x = eddl.Conv(x, 1, [1, 1])

    return x


def main(args):
    eddl.download_drive()

    in_1 = eddl.Input([3, 584, 584])
    in_2 = eddl.Input([1, 584, 584])
    layer = eddl.Concat([in_1, in_2])

    layer = eddl.RandomCropScale(layer, [0.9, 1.0])
    layer = eddl.CenteredCrop(layer, [512, 512])
    img = eddl.Select(layer, ["0:3"])
    mask = eddl.Select(layer, ["3"])

    # DA net
    danet = eddl.Model([in_1, in_2], [])
    eddl.build(danet)
    if args.gpu:
        eddl.toGPU(danet, mem="low_mem")
    eddl.summary(danet)

    # SegNet
    in_ = eddl.Input([3, 512, 512])
    out = eddl.Sigmoid(UNetWithPadding(in_))
    segnet = eddl.Model([in_], [out])
    eddl.build(
        segnet,
        eddl.adam(0.00001),  # Optimizer
        ["mse"],  # Losses
        ["mse"],  # Metrics
        eddl.CS_GPU() if args.gpu else eddl.CS_CPU()
    )
    eddl.summary(segnet)

    print("Reading training data")
    x_train_f = Tensor.load_uint8_t("drive_x.npy")
    x_train = Tensor.permute(x_train_f, [0, 3, 1, 2])
    x_train.info()
    eddlT.div_(x_train, 255.0)

    print("Reading test data")
    y_train = Tensor.load_uint8_t("drive_y.npy")
    y_train.info()
    eddlT.reshape_(y_train, [20, 1, 584, 584])
    eddlT.div_(y_train, 255.0)

    xbatch = eddlT.create([args.batch_size, 3, 584, 584])
    ybatch = eddlT.create([args.batch_size, 1, 584, 584])

    print("Starting training")
    for i in range(args.epochs):
        print("\nEpoch %d/%d" % (i + 1, args.epochs))
        eddl.reset_loss(segnet)
        for j in range(args.num_batches):
            eddl.next_batch([x_train, y_train], [xbatch, ybatch])
            # DA net
            eddl.forward(danet, [xbatch, ybatch])
            xbatch_da = eddl.getTensor(img)
            ybatch_da = eddl.getTensor(mask)
            # SegNet
            eddl.train_batch(segnet, [xbatch_da], [ybatch_da])
            eddl.print_loss(segnet, j)
            if i == args.epochs - 1:
                yout = eddlT.select(eddl.getTensor(out), 0)
                eddlT.save(yout, "./out_%d.jpg" % j)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1)
    parser.add_argument("--num-batches", type=int, metavar="INT", default=5)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
