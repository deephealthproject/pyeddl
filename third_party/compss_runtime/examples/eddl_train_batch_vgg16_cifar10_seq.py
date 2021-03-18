"""\
TRAIN_BATCH example.
"""
import argparse
import platform
import sys
from timeit import default_timer as timer

import numpy as np
import pyeddl._core.eddl as eddl
from pyeddl.tensor import Tensor as eddlT

from models import VGG16
from net_utils import individualParamsSave


def main(args):
    print("E: ", platform.uname())

    eddl.download_cifar10()

    start_time = timer()

    num_workers = args.num_workers
    num_epochs = args.num_epochs
    workers_batch_size = args.workers_batch_size
    num_epochs_for_param_sync = args.num_epochs_for_param_sync
    max_num_async_epochs = args.max_num_async_epochs

    num_classes = 10

    # Model that works
    in_ = eddl.Input([3, 32, 32])
    out = VGG16(in_, num_classes)
    net = eddl.Model([in_], [out])

    eddl.summary(net)

    eddl.build(
        net,
        eddl.sgd(0.001, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU(),
        True
    )

    x_train = eddlT.load("cifar_trX.bin")
    y_train = eddlT.load("cifar_trY.bin")
    x_test = eddlT.load("cifar_tsX.bin")
    y_test = eddlT.load("cifar_tsY.bin")

    eddlT.div_(x_train, 255.0)
    eddlT.div_(x_test, 255.0)

    # Model training
    print("Model training...")
    print("Number of epochs: ", num_epochs)
    print("Number of epochs for parameter syncronization: ", num_epochs_for_param_sync)
    
    s = eddlT.getShape(x_train)
    num_batches = s[0] // workers_batch_size

    for i in range(num_epochs):
        eddl.reset_loss(net)
        print("Epoch %d/%d (%d batches)" % (i + 1, num_epochs, num_batches))
        for j in range(num_batches):
            indices = np.random.randint(0, s[0], workers_batch_size)
            eddl.train_batch(net, [x_train], [y_train], indices)
            print("Termina batch")
            eddl.print_loss(net, j)
            print()


    end_time = timer()
    final_time = end_time - start_time

    print("Elapsed time: ", final_time, " seconds")

    # Model evaluation
    #p = net_parametersToNumpy(net.getParameters())
    #print("Freq: ", getFreqFromParameters(p))

    individualParamsSave(p, "async" + str(max_num_async_epochs) + ".txt")
    print("Evaluating model against train set")
    #eddl.evaluate(net, [x_train], [y_train])

    print("Evaluating model against test set")
    eddl.evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_workers", type=int, metavar="INT", default=4)
    parser.add_argument("--num_epochs", type=int, metavar="INT", default=1)
    parser.add_argument("--num_epochs_for_param_sync", type=int, metavar="INT", default=1)
    parser.add_argument("--max_num_async_epochs", type=int, metavar="INT", default=1)
    parser.add_argument("--workers_batch_size", type=int, metavar="INT", default=250)
    #parser.add_argument("--epochs", type=int, metavar="INT", default=4)
    #parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    #parser.add_argument("--gpu", action="store_true")

    main(parser.parse_args(sys.argv[1:]))
