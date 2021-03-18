"""\
TRAIN_BATCH example.
"""
import argparse
import platform
import sys
from timeit import default_timer as timer

import pyeddl._core.eddl as eddl
from pyeddl.tensor import Tensor as eddlT

import eddl_compss_distributed_api as compss_api
from cvars import *
from eddl_array import array
from models import VGG16
from shuffle import global_shuffle


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

    compss_api.build(
        net,
        eddl.sgd(CVAR_SGD1, CVAR_SGD2),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU(),
        True
    )

    x_train = eddlT.load(CVAR_DATASET_PATH + "cifar_trX.bin")
    y_train = eddlT.load(CVAR_DATASET_PATH + "cifar_trY.bin")
    x_test = eddlT.load(CVAR_DATASET_PATH + "cifar_tsX.bin")
    y_test = eddlT.load(CVAR_DATASET_PATH + "cifar_tsY.bin")

    # Distribute data
    train_images_per_worker = int(eddlT.getShape(x_train)[0] / num_workers)
    x_train_dist = array(x_train, train_images_per_worker)
    y_train_dist = array(y_train, train_images_per_worker)

    eddlT.div_(x_train, 255.0)
    eddlT.div_(x_test, 255.0)

    # Model training
    print("Model training...")
    print("Number of epochs: ", num_epochs)
    print("Number of epochs for parameter syncronization: ", num_epochs_for_param_sync)
    
    for _ in range(0, 1):
        for i in range(0, num_epochs):

            print("Shuffling...")
            x_train_dist, y_train_dist = global_shuffle(x_train_dist, y_train_dist)

            #start_epoch = num_epochs_for_param_sync * i + 1
            #end_epoch = start_epoch + num_epochs_for_param_sync - 1

            #print("Training epochs [", start_epoch, " - ", end_epoch, "] ...")
            print("Training epoch: ", i)
            compss_api.train_batch(
                net,
                x_train_dist,
                y_train_dist,
                num_workers,
                num_epochs_for_param_sync,
                workers_batch_size)

            # Model evaluation
            import sys

            print("Antes de: ")
            sys.stdout.flush()

            p = net.getParameters()
            print("despues ")
            sys.stdout.flush()

            #p = net_parametersToNumpy(p)
            #print("Freq: ", getFreqFromParameters(p))
            sys.stdout.flush()

            #individualParamsSave(p, "sync" + str(num_epochs) + "-epoch" + str(i) + ".txt")

            #print("Evaluating model against train set")
            #eddl.evaluate(net, [x_train], [y_train])

            #print("Evaluating model against test set")
            #eddl.evaluate(net, [x_test], [y_test])

    end_time = timer()
    final_time = end_time - start_time

    print("Elapsed time: ", final_time, " seconds")

    # Model evaluation
    #p = net_parametersToNumpy(net.getParameters())
    #print("Freq: ", getFreqFromParameters(p))

    #individualParamsSave(p, "async" + str(max_num_async_epochs) + ".txt")
    print("Evaluating model against train set")
    eddl.evaluate(net, [x_train], [y_train])

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
