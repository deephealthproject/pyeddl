"""\
TRAIN_BATCH example.
"""
import argparse
import platform
import sys
from timeit import default_timer as timer

import pyeddl._core.eddl as eddl
from pycompss.api.api import compss_wait_on
from pyeddl.tensor import Tensor as eddlT

import eddl_compss_distributed_api as compss_api
import loader
from cvars import *
from eddl_array import apply, paired_partition
from models import LeNet
from net_utils import getFreqFromParameters
from net_utils import individualParamsSave
from net_utils import net_parametersToNumpy
from shuffle import global_shuffle
from utils import download_cifar100_npy


def main(args):
    print("E: ", platform.uname())

    download_cifar100_npy()

    start_time = timer()

    num_workers = args.num_workers
    num_epochs = args.num_epochs
    workers_batch_size = args.workers_batch_size
    num_epochs_for_param_sync = args.num_epochs_for_param_sync
    max_num_async_epochs = args.max_num_async_epochs
    input_file = args.input_file
    output_file = args.output_file

    num_classes = 100

    in_ = eddl.Input([3, 32, 32])
    out = LeNet(in_, num_classes)
    net = eddl.Model([in_], [out])

    eddl.summary(net)

    compss_api.build(
        net,
        eddl.sgd(CVAR_SGD1, CVAR_SGD2),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU(),
        True if input_file == "" else False
    )

    # With data loaders, data are directly loaded in a distributed fashion
    x_train = loader.load_npy(CVAR_DATASET_PATH + "cifar100_trX.npy", num_workers)
    y_train = loader.load_npy(CVAR_DATASET_PATH + "cifar100_trY.npy", num_workers)
    x_test = loader.load_npy(CVAR_DATASET_PATH + "cifar100_tsX.npy", num_workers)
    y_test = loader.load_npy(CVAR_DATASET_PATH + "cifar100_tsY.npy", num_workers)

    # In this case, since local data do not exist, all functions must be applied by COMPSs
    x_train = apply(eddlT.div_, x_train, 255.0)
    x_text = apply(eddlT.div_, x_test, 255.0)

    # Model training
    print("Model training...")
    print("Number of epochs: ", num_epochs)
    print("Number of epochs for parameter syncronization: ", num_epochs_for_param_sync)

    for _ in range(0, 1):
        for i in range(0, num_epochs):
            epoch_start_time = timer()

            print("Shuffling...")
            x_train, y_train = global_shuffle(x_train, y_train)

            # start_epoch = num_epochs_for_param_sync * i + 1
            # end_epoch = start_epoch + num_epochs_for_param_sync - 1

            # print("Training epochs [", start_epoch, " - ", end_epoch, "] ...")
            print("Training epoch: ", i)
            compss_api.train_batch(
                net,
                x_train,
                y_train,
                num_workers,
                num_epochs_for_param_sync,
                workers_batch_size)

            # Model evaluation
            import sys

            p = net_parametersToNumpy(net.getParameters())
            # print("Los final weights en main son estos: ", p)

            epoch_end_time = timer()
            print("Elapsed time for epoch ", str(i), ": ", str(epoch_end_time - epoch_start_time), " seconds")

            print("Freq: ", getFreqFromParameters(p))
            sys.stdout.flush()

            print("Param saving")
            individualParamsSave(p, "./res_cifar100_lenet_sync/lenet_params_sync_from_0-epoch" + str(i) + ".txt")

            print("Net saving")
            eddl.save_net_to_onnx_file(net, "./res_cifar100_lenet_sync/lenet_net_sync_from_0-epoch" + str(i) + ".onnx")

            # SUPPORT FOR DISTRIBUTRED EVALUATION IS REQUIRED HERE
            # print("Evaluating model against train set")
            # eddl.evaluate(net, [x_train], [y_train])

            # print("Evaluating model against test set")
            # compss_api.eddl.evaluate(net, [x_test], [y_test])

    end_time = timer()
    final_time = end_time - start_time

    print("Elapsed time: ", final_time, " seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_workers", type=int, metavar="INT", default=4)
    parser.add_argument("--num_epochs", type=int, metavar="INT", default=1)
    parser.add_argument("--num_epochs_for_param_sync", type=int, metavar="INT", default=1)
    parser.add_argument("--max_num_async_epochs", type=int, metavar="INT", default=1)
    parser.add_argument("--workers_batch_size", type=int, metavar="INT", default=250)
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    # parser.add_argument("--epochs", type=int, metavar="INT", default=4)
    # parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    # parser.add_argument("--gpu", action="store_true")

    main(parser.parse_args(sys.argv[1:]))
