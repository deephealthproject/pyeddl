"""\
ResNet18 for CIFAR10.
"""

import argparse
import sys

import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT


def ResBlock(layer, filters, nconv, half):
    in_ = layer
    strides = [2, 2] if half else [1, 1]
    layer = eddl.ReLu(eddl.Conv(layer, filters, [3, 3], strides))
    for i in range(nconv - 1):
        layer = eddl.ReLu(eddl.Conv(layer, filters, [3, 3], [1, 1]))
    if (half):
        return eddl.Sum(eddl.Conv(in_, filters, [1, 1], [2, 2]), layer)
    else:
        return eddl.Sum(layer, in_)


def main(args):
    eddl.download_cifar10()

    num_classes = 10

    in_ = eddl.Input([3, 32, 32])

    layer = in_
    layer = eddl.ReLu(eddl.Conv(layer, 64, [3, 3], [1, 1]))
    layer = ResBlock(layer, 64, 2, True)
    layer = ResBlock(layer, 64, 2, False)
    layer = ResBlock(layer, 128, 2, True)
    layer = ResBlock(layer, 128, 2, False)
    layer = ResBlock(layer, 256, 2, True)
    layer = ResBlock(layer, 256, 2, False)
    layer = ResBlock(layer, 512, 2, True)
    layer = ResBlock(layer, 512, 2, False)
    layer = eddl.Reshape(layer, [-1])
    layer = eddl.Activation(eddl.Dense(layer, 512), "relu")

    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU([1]) if args.gpu else eddl.CS_CPU()
    )

    eddl.summary(net)
    eddl.plot(net, "model.pdf")

    x_train = eddlT.load("cifar_trX.bin")
    y_train = eddlT.load("cifar_trY.bin")
    eddlT.div_(x_train, 255.0)

    x_test = eddlT.load("cifar_tsX.bin")
    y_test = eddlT.load("cifar_tsY.bin")
    eddlT.div_(x_test, 255.0)

    for i in range(args.epochs):
        eddl.fit(net, [x_train], [y_train], args.batch_size, 1)
        eddl.evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=100)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
