"""\
TRAIN_BATCH example.
"""

import argparse
import sys

import numpy as np
from pyeddl.api import (
    Input, Activation, Dense, Model, sgd, CS_CPU, build, T_load, div,
    evaluate, CS_GPU, resize_model, set_mode, TRMODE, train_batch
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
    x_test = T_load("tsX.bin")
    y_test = T_load("tsY.bin")

    div(x_train, 255.0)
    div(x_test, 255.0)

    tin = [x_train.data]
    tout = [y_train.data]
    num_samples = tin[0].shape[0]
    num_batches = num_samples // batch_size
    resize_model(net, batch_size)
    set_mode(net, TRMODE)
    for i in range(epochs):
        for j in range(num_batches):
            print("Epoch %d/%d (batch %d/%d)" %
                  (i + 1, epochs, j + 1, num_batches))
            indices = np.random.randint(0, num_samples, batch_size)
            train_batch(net, tin, tout, indices)
    evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
