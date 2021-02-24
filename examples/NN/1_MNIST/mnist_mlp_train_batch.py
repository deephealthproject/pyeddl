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
Basic MLP for MNIST with batch training and evaluation.
"""

import argparse
import sys

import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor


MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")


def main(args):
    eddl.download_mnist()

    num_classes = 10

    in_ = eddl.Input([784])
    layer = in_
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    out = eddl.Softmax(eddl.Dense(layer, num_classes))
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.sgd(0.001, 0.9),
        ["softmax_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )

    eddl.summary(net)
    eddl.plot(net, "model.pdf")
    eddl.setlogfile(net, "mnist")

    x_train = Tensor.load("mnist_trX.bin")
    y_train = Tensor.load("mnist_trY.bin")
    x_test = Tensor.load("mnist_tsX.bin")
    y_test = Tensor.load("mnist_tsY.bin")
    if args.small:
        x_train = x_train.select([":6000"])
        y_train = y_train.select([":6000"])
        x_test = x_test.select([":1000"])
        y_test = y_test.select([":1000"])

    x_train.div_(255.0)
    x_test.div_(255.0)

    s = x_train.shape
    num_batches = s[0] // args.batch_size
    for i in range(args.epochs):
        eddl.reset_loss(net)
        print("Epoch %d/%d (%d batches)" % (i + 1, args.epochs, num_batches))
        for j in range(num_batches):
            indices = np.random.randint(0, s[0], args.batch_size)
            eddl.train_batch(net, [x_train], [y_train], indices)

    losses1 = eddl.get_losses(net)
    metrics1 = eddl.get_metrics(net)
    for l, m in zip(losses1, metrics1):
        print("Loss: %.6f\tMetric: %.6f" % (l, m))

    s = x_test.shape
    num_batches = s[0] // args.batch_size
    for j in range(num_batches):
        indices = np.arange(j * args.batch_size,
                            j * args.batch_size + args.batch_size)
        eddl.eval_batch(net, [x_test], [y_test], indices)

    losses2 = eddl.get_losses(net)
    metrics2 = eddl.get_metrics(net)
    for l, m in zip(losses2, metrics2):
        print("Loss: %.6f\tMetric: %.6f" % (l, m))

    last_batch_size = s[0] % args.batch_size
    if last_batch_size:
        indices = np.arange(j * args.batch_size,
                            j * args.batch_size + args.batch_size)
        eddl.eval_batch(net, [x_test], [y_test], indices)

    losses3 = eddl.get_losses(net)
    metrics3 = eddl.get_metrics(net)
    for l, m in zip(losses3, metrics3):
        print("Loss: %.6f\tMetric: %.6f" % (l, m))

    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES),
                        choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
