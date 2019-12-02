"""\
LOAD_SAVE example.
"""

import argparse
import sys

import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT


def main(args):
    eddl.download_mnist()

    num_classes = 10

    in_ = eddl.Input([784])
    layer = in_
    layer = eddl.Activation(eddl.Dense(layer, 1024), "relu")
    layer = eddl.Activation(eddl.Dense(layer, 1024), "relu")
    layer = eddl.Activation(eddl.Dense(layer, 1024), "relu")
    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU([1]) if args.gpu else eddl.CS_CPU(4)
    )

    eddl.summary(net)
    eddl.plot(net, "model.pdf")

    x_train = eddlT.load("trX.bin")
    y_train = eddlT.load("trY.bin")

    eddlT.div_(x_train, 255.0)

    eddl.save(net, "model1.bin", "bin")

    eddl.fit(net, [x_train], [y_train], args.batch_size, args.epochs)

    eddl.load(net, "model1.bin", "bin")

    eddl.fit(net, [x_train], [y_train], args.batch_size, args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=1)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
