"""\
CONV example.
"""

import argparse
import sys

import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT


def Block(layer, filters, kernel_size, strides):
    return eddl.MaxPool(
        eddl.BatchNormalization(
            eddl.Activation(
                eddl.L1(
                    eddl.Conv(layer, filters, kernel_size, strides),
                    0.0001
                ),
                "relu"
            )
        ),
        [2, 2]
    )


def main(args):
    eddl.download_mnist()

    num_classes = 10

    in_ = eddl.Input([784])
    layer = in_
    layer = eddl.Reshape(layer, [1, 28, 28])
    layer = eddl.UpSampling(layer, [2, 2])
    layer = eddl.MaxPool(layer, [2, 2])
    layer = Block(layer, 16, [3, 3], [1, 1])
    layer = Block(layer, 32, [3, 3], [1, 1])
    layer = Block(layer, 64, [3, 3], [1, 1])
    layer = Block(layer, 128, [3, 3], [1, 1])
    layer = eddl.Reshape(layer, [-1])
    layer = eddl.Activation(eddl.Dense(layer, 64), "relu")
    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")

    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU([1]) if args.gpu else eddl.CS_CPU(4)
    )

    print(eddl.summary(net))

    x_train = eddlT.load("trX.bin")
    y_train = eddlT.load("trY.bin")

    eddlT.div_(x_train, 255.0)

    eddl.fit(net, [x_train], [y_train], args.batch_size, args.epochs)
    eddl.evaluate(net, [x_train], [y_train])

    x_test = eddlT.load("tsX.bin")
    y_test = eddlT.load("tsY.bin")

    eddlT.div_(x_test, 255.0)
    eddl.evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
