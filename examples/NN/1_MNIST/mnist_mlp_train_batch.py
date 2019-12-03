"""\
Basic MLP for MNIST with batch training and evaluation.
"""

import argparse
import sys

import numpy as np
import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT


def main(args):
    eddl.download_mnist()

    num_classes = 10

    in_ = eddl.Input([784])
    layer = in_
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
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
    eddl.setlogfile(net, "mnist")

    x_train = eddlT.load("trX.bin")
    y_train = eddlT.load("trY.bin")
    x_test = eddlT.load("tsX.bin")
    y_test = eddlT.load("tsY.bin")

    eddlT.div_(x_train, 255.0)
    eddlT.div_(x_test, 255.0)

    s = eddlT.getShape(x_train)
    num_batches = s[0] // args.batch_size
    for i in range(args.epochs):
        eddl.reset_loss(net)
        print("Epoch %d/%d (%d batches)" % (i + 1, args.epochs, num_batches))
        for j in range(num_batches):
            indices = np.random.randint(0, s[0], args.batch_size)
            eddl.train_batch(net, [x_train], [y_train], indices)
            eddl.print_loss(net, j)
            print()

    s = eddlT.getShape(x_test)
    num_batches = s[0] // args.batch_size
    for j in range(num_batches):
        indices = np.arange(j * args.batch_size,
                            j * args.batch_size + args.batch_size)
        eddl.eval_batch(net, [x_test], [y_test], indices)
        eddl.print_loss(net, j)
        print()

    last_batch_size = s[0] % args.batch_size
    if last_batch_size:
        indices = np.arange(j * args.batch_size,
                            j * args.batch_size + args.batch_size)
        eddl.eval_batch(net, [x_test], [y_test], indices)
        eddl.print_loss(net, j)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
