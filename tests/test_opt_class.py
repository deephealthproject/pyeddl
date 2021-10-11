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

import pytest
import pyeddl.eddl as eddl
from pyeddl._core import SGD, Adam, AdaDelta, Adagrad, Adamax, Nadam, RMSProp

OPT_CLASSES = SGD, Adam, AdaDelta, Adagrad, Adamax, Nadam, RMSProp


@pytest.mark.parametrize("opt_cls", OPT_CLASSES)
def test_build_net(opt_cls):
    num_classes = 10
    in_ = eddl.Input([784])
    layer = in_
    layer = eddl.LeakyReLu(eddl.Dense(layer, 1024))
    layer = eddl.LeakyReLu(eddl.Dense(layer, 1024))
    layer = eddl.LeakyReLu(eddl.Dense(layer, 1024))
    out = eddl.Softmax(eddl.Dense(layer, num_classes), -1)
    net = eddl.Model([in_], [out])
    eddl.build(
        net,
        opt_cls(0.01),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU(mem="low_mem")
    )
