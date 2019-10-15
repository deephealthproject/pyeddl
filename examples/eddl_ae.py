"""\
AE example.
"""

import argparse
import sys

from pyeddl.api import (
    Input, Activation, Dense, Model, sgd, CS_CPU, build, T_load, div, fit,
    CS_GPU
)
from pyeddl.utils import download_mnist


def main(args):
    download_mnist()

    epochs = args.epochs
    batch_size = args.batch_size

    in_ = Input([784])
    layer = in_
    layer = Activation(Dense(layer, 256), "relu")
    layer = Activation(Dense(layer, 128), "relu")
    layer = Activation(Dense(layer, 64), "relu")
    layer = Activation(Dense(layer, 128), "relu")
    layer = Activation(Dense(layer, 256), "relu")
    out = Dense(layer, 784)
    net = Model([in_], [out])

    build(
        net,
        sgd(0.001, 0.9),
        ["mean_squared_error"],
        ["mean_squared_error"],
        CS_GPU([1]) if args.gpu else CS_CPU(4)
    )

    print(net.summary())

    x_train = T_load("trX.bin")
    div(x_train, 255.0)
    fit(net, [x_train], [x_train], batch_size, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=100)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
