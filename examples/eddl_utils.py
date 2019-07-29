import os
import subprocess
import pyeddl._core as pyeddl


DATA_URLS = [
    "https://www.dropbox.com/s/khrb3th2z6owd9t/trX.bin",
    "https://www.dropbox.com/s/m82hmmrg46kcugp/trY.bin",
    "https://www.dropbox.com/s/7psutd4m4wna2d5/tsX.bin",
    "https://www.dropbox.com/s/q0tnbjvaenb4tjs/tsY.bin",
]


def download_mnist():
    for url in DATA_URLS:
        fn = url.rsplit("/", 1)[-1]
        if not os.path.exists(fn):
            print("getting ", url)
            subprocess.check_call("wget %s" % url, shell=True)


def loss_func(fname):
    if fname == "mse" or fname == "mean_squared_error":
        return pyeddl.LMeanSquaredError()
    elif fname == "cross_entropy":
        return pyeddl.LCrossEntropy()
    elif (fname == "soft_cross_entropy"):
        return pyeddl.LSoftCrossEntropy()
    else:
        return None


def metric_func(fname):
    if fname == "mse" or fname == "mean_squared_error":
        return pyeddl.MMeanSquaredError()
    elif fname == "categorical_accuracy" or fname == "accuracy":
        return pyeddl.MCategoricalAccuracy()
    else:
        return None
