"""\
LOAD_SAVE example.
"""

import argparse
import sys

from pyeddl.api import (
    Input, Activation, Dense, Model, sgd, CS_CPU, build, T_load, div, fit,
    save, load, CS_GPU
)
from pyeddl.utils import download_mnist


def main(args):
    download_mnist()

    epochs = args.epochs
    batch_size = args.batch_size
    num_classes = 10

    in_ = Input([784])
    layer = in_
    layer = Activation(Dense(layer, 1024), "relu")
    layer = Activation(Dense(layer, 1024), "relu")
    layer = Activation(Dense(layer, 1024), "relu")
    out = Activation(Dense(layer, num_classes), "softmax")
    net = Model([in_], [out])
    print(net.summary())

    build(
        net,
        sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        CS_GPU([1]) if args.gpu else CS_CPU(4)
    )

    x_train = T_load("trX.bin")
    y_train = T_load("trY.bin")

    div(x_train, 255.0)

    save(net, "model1.bin")

    fit(net, [x_train], [y_train], batch_size, epochs)

    load(net, "model1.bin")

    fit(net, [x_train], [y_train], batch_size, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=1)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
