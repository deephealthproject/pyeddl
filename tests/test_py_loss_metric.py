# Copyright (c) 2019-2020, CRS4
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
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
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import pytest
import pyeddl.eddl as eddl
import pyeddl.eddlT as eddlT
from pyeddl._core import Loss, Metric


class MSELoss(Loss):

    def __init__(self):
        Loss.__init__(self, "py_mean_squared_error")

    def delta(self, t, y, d):
        eddlT.copyTensor(eddlT.sub(y, t), d)
        eddlT.div_(d, eddlT.getShape(t)[0])

    def value(self, t, y):
        size = t.size / eddlT.getShape(t)[0]
        aux = eddlT.add(t, eddlT.neg(y))
        aux = eddlT.mult(aux, aux)
        return aux.sum() / size


class MSEMetric(Metric):

    def __init__(self):
        Metric.__init__(self, "py_mean_squared_error")

    def value(self, t, y):
        size = t.size / eddlT.getShape(t)[0]
        aux = eddlT.add(t, eddlT.neg(y))
        aux = eddlT.mult(aux, aux)
        return aux.sum() / size


class CategoricalAccuracy(Metric):

    def __init__(self):
        Metric.__init__(self, "py_categorical_accuracy")

    def value(self, t, y):
        a = eddlT.getdata(t)
        b = eddlT.getdata(y)
        return (np.argmax(a, axis=-1) == np.argmax(b, axis=-1)).sum()


def test_py_metric():
    shape = [8, 10]
    a = np.random.random(shape).astype(np.float32)
    b = np.random.random(shape).astype(np.float32)
    t, y = eddlT.create(a), eddlT.create(b)
    v = MSEMetric().value(t, y)
    exp_v = eddl.getMetric("mse").value(t, y)
    assert v == pytest.approx(exp_v)
    v = CategoricalAccuracy().value(t, y)
    exp_v = eddl.getMetric("categorical_accuracy").value(t, y)
    assert v == pytest.approx(exp_v)


def test_py_loss():
    shape = [8, 10]
    a = np.random.random(shape).astype(np.float32)
    b = np.random.random(shape).astype(np.float32)
    t, y = eddlT.create(a), eddlT.create(b)
    z = eddlT.create(shape)
    exp_z = eddlT.create(shape)
    py_mse_loss = MSELoss()
    mse_loss = eddl.getLoss("mse")
    mse_loss.delta(t, y, exp_z)
    py_mse_loss.delta(t, y, z)
    c = eddlT.getdata(z)
    exp_c = eddlT.getdata(exp_z)
    assert np.array_equal(c, exp_c)
    v = py_mse_loss.value(t, y)
    exp_v = mse_loss.value(t, y)
    assert v == pytest.approx(exp_v)
