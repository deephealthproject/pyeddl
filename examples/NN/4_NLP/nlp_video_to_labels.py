# Copyright (c) 2019-2021 CRS4
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
Video to labels example.
"""

import argparse
import sys

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor


MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")


def main(args):

    size = 256 // 2

    # Conv3D expects (B, C, dim1, dim2, dim3)
    in_ = eddl.Input([3, 10, size, size])
    layer = in_
    layer = eddl.MaxPool3D(eddl.ReLu(eddl.Conv3D(
        layer, 4, [1, 3, 3], [1, 1, 1], "same"
    )), [1, 2, 2], [1, 2, 2], "same")
    layer = eddl.MaxPool3D(eddl.ReLu(eddl.Conv3D(
        layer, 8, [1, 3, 3], [1, 1, 1], "same"
    )), [1, 2, 2], [1, 2, 2], "same")
    layer = eddl.MaxPool3D(eddl.ReLu(eddl.Conv3D(
        layer, 16, [1, 3, 3], [1, 1, 1], "same"
    )), [1, 2, 2], [1, 2, 2], "same")
    layer = eddl.GlobalMaxPool3D(layer)
    layer = eddl.Reshape(layer, [-1])
    layer = eddl.LSTM(layer, 128)
    layer = eddl.Dense(layer, 100)
    layer = eddl.ReLu(layer)
    layer = eddl.Dense(layer, 2)
    out = eddl.ReLu(layer)
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.adam(),
        ["mse"],
        ["mse"],
        eddl.CS_GPU(mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )
    eddl.summary(net)

    seqImages = Tensor.randu([32, 10, 3, 10, size, size])
    seqLabels = Tensor.randu([32, 7, 2])
    eddl.fit(net, [seqImages], [seqLabels], 4, 2 if args.small else 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES),
                        choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args(sys.argv[1:]))
