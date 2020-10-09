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
MNIST encoder-decoder with user-defined losses.
"""

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor


# l2_loss
def mse_loss(inputs):
    diff = eddl.Diff(inputs[0], inputs[1])
    return eddl.Mult(diff, diff)


# Dice loss image-level
def dice_loss_img(inputs):
    num = eddl.Mult(2, eddl.ReduceSum(
        eddl.Mult(inputs[0], inputs[1]), [0, 1, 2]
    ))
    den = eddl.ReduceSum(eddl.Sum(inputs[0], inputs[1]), [0, 1, 2])
    return eddl.Diff(1.0, eddl.Div(num, den))


# Dice loss pixel-level
def dice_loss_pixel(inputs):
    num = eddl.Mult(2, eddl.ReduceSum(eddl.Mult(inputs[0], inputs[1]), [0]))
    den = eddl.ReduceSum(eddl.Sum(inputs[0], inputs[1]), [0])
    num = eddl.Sum(num, 1)
    den = eddl.Sum(den, 1)
    return eddl.Diff(1.0, eddl.Div(num, den))


def main(args):
    eddl.download_mnist()

    in_ = eddl.Input([784])
    target = eddl.Reshape(in_, [1, 28, 28])
    layer = in_
    layer = eddl.Reshape(layer, [1, 28, 28])
    layer = eddl.ReLu(eddl.Conv(layer, 8, [3, 3]))
    layer = eddl.ReLu(eddl.Conv(layer, 16, [3, 3]))
    layer = eddl.ReLu(eddl.Conv(layer, 8, [3, 3]))
    out = eddl.Sigmoid(eddl.Conv(layer, 1, [3, 3]))
    net = eddl.Model([in_], [])

    eddl.build(
        net,
        eddl.adam(0.001),
        [],
        [],
        eddl.CS_GPU() if args.gpu else eddl.CS_CPU()
    )
    eddl.summary(net)

    x_train = Tensor.load("mnist_trX.bin")
    if args.small:
        x_train = x_train.select([":6000"])
    x_train.div_(255.0)

    mse = eddl.newloss(mse_loss, [out, target], "mse_loss")
    dicei = eddl.newloss(dice_loss_img, [out, target], "dice_loss_img")
    dicep = eddl.newloss(dice_loss_pixel, [out, target], "dice_loss_pixel")

    batch = Tensor([args.batch_size, 784])
    num_batches = x_train.shape[0] // args.batch_size
    for i in range(args.epochs):
        print("Epoch %d/%d (%d batches)" % (i + 1, args.epochs, num_batches))
        diceploss = 0.0
        diceiloss = 0.0
        mseloss = 0
        for j in range(num_batches):
            print("Batch %d " % j, end="", flush=True)
            eddl.next_batch([x_train], [batch])
            eddl.zeroGrads(net)
            eddl.forward(net, [batch])
            diceploss += eddl.compute_loss(dicep) / args.batch_size
            print("diceploss = %.6f " % (diceploss / (j + 1)), end="")
            diceiloss += eddl.compute_loss(dicei) / args.batch_size
            print("diceiloss = %.6f " % (diceiloss / (j + 1)), end="")
            mseloss += eddl.compute_loss(mse) / args.batch_size
            print("mseloss = %.6f\r" % (mseloss / (j + 1)), end="")
            eddl.optimize(dicep)
            eddl.update(net)
        print()
    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--small", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
