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
import pyeddl._core.eddl as eddl_core
import pyeddl.eddl as eddl_py


LAYER_ATTRS = [
    "name",
    "input",
    "output",
    "target",
    "delta",
    "orig",
    "sorig",
    "net",
    "trainable",
    "mem_level",
    "isrecurrent",
    "isshared",
    "iscloned",
    "isnorm",
    "isdecoder",
    "distributed_training",
    "params",
    "gradients",
    "states",
    "delta_states",
    "acc_gradients",
    "parent",
    "child",
    "clones",
    "reg",
    "init",
    "mode",
    "dev",
    "lin",
    "lout",
    "delta_bp",
    "detached",
    "do_deletes",
    "verbosity_level",
]

LOSS_ATTRS = [
    "name",
]

METRIC_ATTRS = [
    "name",
]

NET_ATTRS = [
    "name",
    "dev",
    "batch_size",
    "tr_batches",
    "inferenced_samples",
    "trmode",
    "mem_level",
    "verbosity_level",
    "onnx_pretrained",
    "isrecurrent",
    "isbuild",
    "isdecoder",
    "isencoder",
    "isresized",
    "decoder_teacher_training",
    "decsize",
    "devsel",
    "cs",
    "do_compserv_delete",
    "layers",
    "layersf",
    "layersb",
    "lin",
    "din",
    "lout",
    "vfts",
    "vbts",
    "netinput",
    "losses",
    "metrics",
    "fiterr",
    "total_loss",
    "total_metric",
    "optimizer",
    "do_optimizer_delete",
    "snets",
    "rnet",
    "Xs",
    "Ys",
]

NETLOSS_ATTRS = [
    "name",
    "value",
    "graph",
    "input",
    "ginput",
    "fout",
]

OPTIMIZER_ATTRS = [
    "name",
    "layers",
    "isshared",
    "clip_val",
    "orig",
]


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_layer(eddl):
    in_ = eddl.Input([16])
    for a in LAYER_ATTRS:
        getattr(in_, a)


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_loss(eddl):
    loss = eddl.getLoss("mse")
    for a in LOSS_ATTRS:
        getattr(loss, a)


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_metric(eddl):
    metric = eddl.getMetric("mse")
    for a in METRIC_ATTRS:
        getattr(metric, a)


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_net(eddl):
    in_ = eddl.Input([32])
    layer = eddl.LeakyReLu(eddl.Dense(in_, 16))
    out = eddl.Softmax(eddl.Dense(layer, 10))
    net = eddl.Model([in_], [out])
    for a in NET_ATTRS:
        getattr(net, a)


# For some reason, when eddl is eddl_core and all tests are run at the same
# time, this leads to a segfault in test_eddl.py
@pytest.mark.parametrize("eddl", [eddl_py])
def test_netloss(eddl):

    def mse_loss(inputs):
        diff = eddl.Sub(inputs[0], inputs[1])
        return eddl.Mult(diff, diff)

    in_ = eddl.Input([16])
    target = eddl.Reshape(in_, [1, 4, 4])
    layer = in_
    layer = eddl.Reshape(layer, [1, 4, 4])
    layer = eddl.ReLu(eddl.Conv(layer, 8, [3, 3]))
    out = eddl.Sigmoid(eddl.Conv(layer, 1, [3, 3]))
    net = eddl.Model([in_], [])
    eddl.build(net, eddl.adam(0.001), [], [], eddl.CS_CPU())
    mse = eddl.newloss(mse_loss, [out, target], "mse_loss")
    for a in NETLOSS_ATTRS:
        getattr(mse, a)


@pytest.mark.parametrize("eddl", [eddl_core, eddl_py])
def test_optimizer(eddl):
    opt = eddl.adam(0.001)
    for a in OPTIMIZER_ATTRS:
        getattr(opt, a)
