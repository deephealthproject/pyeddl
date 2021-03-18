"""\
TRAIN_BATCH example.
"""
import argparse
import platform
import sys
from timeit import default_timer as timer

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor as eddlT

import eddl_compss_distributed_api as compss_api
from cvars import *
from eddl_array import array
from shuffle import global_shuffle

import loader
from eddl_array import apply, paired_partition


def main(args):
    print("E: ", platform.uname())

    eddl.download_mnist()

    start_time = timer()

    num_workers = args.num_workers
    num_epochs = args.num_epochs
    workers_batch_size = args.workers_batch_size
    num_epochs_for_param_sync = args.num_epochs_for_param_sync
    max_num_async_epochs = args.max_num_async_epochs

    num_classes = 10

    # Model that works
    in_ = eddl.Input([784])

    layer = in_
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")
    net = eddl.Model([in_], [out])

    eddl.summary(net)

    compss_api.build(
        net,
        eddl.sgd(CVAR_SGD1, CVAR_SGD2),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU(),
        True
    )

    # With data loaders, data are directly loaded in a distributed fashion
    x_train = loader.load_npy(CVAR_DATASET_PATH + "mnist_trX.npy", num_workers)
    y_train = loader.load_npy(CVAR_DATASET_PATH + "mnist_trY.npy", num_workers)
    x_test = loader.load_npy(CVAR_DATASET_PATH + "mnist_tsX.npy", num_workers)
    y_test = loader.load_npy(CVAR_DATASET_PATH + "mnist_tsY.npy", num_workers)

    # In this case, since local data do not exist, all functions must be applied by COMPSs
    x_train = apply(eddlT.div_, x_train, 255.0)
    print("Cuanto vale x_train: ", x_train)
    x_test = apply(eddlT.div_, x_test, 255.0)

    # Model training
    print("Model training...")
    print("Number of epochs: ", num_epochs)
    print("Number of epochs for parameter syncronization: ", num_epochs_for_param_sync)

    for _ in range(0, 1):
        for i in range(0, num_epochs):
            print("Shuffling...")
            x_train, y_train = global_shuffle(x_train, y_train)

            print("Training epoch: ", i)
            compss_api.train_batch(
                net,
                x_train,
                y_train,
                num_workers,
                num_epochs_for_param_sync,
                workers_batch_size)


    x_test, y_test = global_shuffle(x_test, y_test)
    metrics_losses = compss_api.eval_batch(net, x_test, y_test, num_workers, workers_batch_size)

    print("Losses and metrics: ", metrics_losses)

    end_time = timer()
    final_time = end_time - start_time

    print("Elapsed time: ", final_time, " seconds")

    print("TEST")

    x_test = eddlT.load(CVAR_DATASET_PATH + "mnist_tsX.bin")
    y_test = eddlT.load(CVAR_DATASET_PATH + "mnist_tsY.bin")

    eddlT.div_(x_test, 255.0)

    # Model evaluation
    print("Evaluating model against test set")
    eddl.evaluate(net, [x_test], [y_test])

    losses1 = eddl.get_losses(net)
    metrics1 = eddl.get_metrics(net)
    print("Len: ", len(losses1))
    
    for l, m in zip(losses1, metrics1):
        print("Loss: %.6f\tMetric: %.6f" % (l, m))

    print("FIN TEST")
    
    # Model evaluation
    #print("Evaluating model against test set")
    #eddl.evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_workers", type=int, metavar="INT", default=4)
    parser.add_argument("--num_epochs", type=int, metavar="INT", default=1)
    parser.add_argument("--num_epochs_for_param_sync", type=int, metavar="INT", default=1)
    parser.add_argument("--max_num_async_epochs", type=int, metavar="INT", default=1)
    parser.add_argument("--workers_batch_size", type=int, metavar="INT", default=250)
    # parser.add_argument("--epochs", type=int, metavar="INT", default=4)
    # parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    # parser.add_argument("--gpu", action="store_true")

    main(parser.parse_args(sys.argv[1:]))
