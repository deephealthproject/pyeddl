# Copyright (c) 2019-2020, CRS4
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""\
Import model from file in ONNX format.
"""

import argparse
import os
import sys

import pyeddl.eddl as eddl
import pyeddl.eddlT as eddlT


def main(args):
    if not os.path.isfile(args.input):
        raise RuntimeError("input file '%s' not found" % args.input)

    eddl.download_mnist()

    print("importing net from", args.input)
    net = eddl.import_net_from_onnx_file(args.input)
    print("output size =", len(net.lout))

    eddl.build(
        net,
        eddl.rmsprop(0.01),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU([1]) if args.gpu else eddl.CS_CPU(),
        False  # do not initialize weights to random values
    )

    net.resize(args.batch_size)  # resize manually since we don't use "fit"
    eddl.summary(net)

    x_test = eddlT.load("tsX.bin")
    y_test = eddlT.load("tsY.bin")

    eddlT.div_(x_test, 255.0)

    eddl.evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--input", metavar="STRING",
                        default="trained_model.onnx",
                        help="input path of the serialized model")
    main(parser.parse_args(sys.argv[1:]))
