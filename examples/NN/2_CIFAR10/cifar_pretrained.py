# Copyright (c) 2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""\
CIFAR example from pretrained resnet18.
"""

import argparse
import os
import sys
from urllib.request import urlretrieve

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor


MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")
MODEL_URL = "https://www.dropbox.com/s/tn0d87dr035yhol/resnet18-v1-7.onnx"
MODEL_PATH = "resnet18-v1-7.onnx"


def main(args):
    eddl.download_cifar10()
    if not os.path.isfile(MODEL_PATH):
        print("downloading", os.path.basename(MODEL_PATH))
        urlretrieve(MODEL_URL + "?dl=1", MODEL_PATH)

    net_onnx = eddl.import_net_from_onnx_file(MODEL_PATH)
    # remove last layer
    eddl.removeLayer(net_onnx, "resnetv15_dense0_fwd")

    # create a new graph to adapt the output for CIFAR
    in_ = eddl.Input([512])
    layer = eddl.Dense(in_, 10)
    layer = eddl.Softmax(layer)

    net_adap = eddl.Model([in_], [layer])

    # cat both models
    net = eddl.Model([net_onnx, net_adap])

    eddl.build(
        net,
        eddl.sgd(0.001),
        ["softmax_cross_entropy"],
        ["accuracy"],
        eddl.CS_GPU(mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )
    eddl.summary(net)

    # from "flatten_170" to the beginning
    eddl.setTrainable(net, "flatten_170", False)

    # Load dataset
    x_train = Tensor.load("cifar_trX.bin")
    xs = x_train.select([":1000", ":"])
    xtrain = xs.scale([224, 224])
    xtrain.div(255.0)
    del x_train
    del xs

    y_train = Tensor.load("cifar_trY.bin")
    ytrain = y_train.select([":1000", ":"])

    for i in range(args.epochs):
        eddl.fit(net, [xtrain], [ytrain], args.batch_size, 1)

    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=100)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--small", action="store_true")  # no-op
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES),
                        choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
