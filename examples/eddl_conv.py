"""\
CONV example.
"""

import argparse
import sys

from pyeddl.api import (
    Input, Activation, Dense, Model, sgd, CS_CPU, build, T_load, div, fit,
    evaluate, Reshape, MaxPool, Conv, CS_GPU, BatchNormalization
)
from pyeddl.utils import download_mnist


def Block(layer, filters, kernel_size, strides, gpu=False):
    b = Activation(Conv(layer, filters, kernel_size, strides), "relu")
    if not gpu:
        b = BatchNormalization(b)
    return MaxPool(b, [2, 2])


def main(args):
    download_mnist()

    epochs = args.epochs
    batch_size = args.batch_size
    num_classes = 10

    in_ = Input([784])
    layer = in_

    layer = Reshape(layer, [1, 28, 28])

    layer = Block(layer, 16, [3, 3], [1, 1], gpu=args.gpu)
    layer = Block(layer, 32, [3, 3], [1, 1], gpu=args.gpu)
    layer = Block(layer, 64, [3, 3], [1, 1], gpu=args.gpu)
    layer = Block(layer, 128, [3, 3], [1, 1], gpu=args.gpu)

    layer = Reshape(layer, [-1])

    layer = Activation(Dense(layer, 256), "relu")
    out = Activation(Dense(layer, num_classes), "softmax")

    net = Model([in_], [out])

    build(
        net,
        sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        CS_GPU([1]) if args.gpu else CS_CPU(4)
    )

    print(net.summary())

    X = T_load("trX.bin")
    Y = T_load("trY.bin")

    div(X, 255.0)

    fit(net, [X], [Y], batch_size, epochs)
    evaluate(net, [X], [Y])

    tX = T_load("tsX.bin")
    tY = T_load("tsY.bin")

    div(tX, 255.0)

    evaluate(net, [tX], [tY])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=5)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=100)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
