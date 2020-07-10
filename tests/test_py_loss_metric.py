# Copyright (c) 2019-2020 CRS4
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

import numpy as np
import pytest
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from pyeddl._core import Loss, Metric


class MSELoss(Loss):

    def __init__(self):
        Loss.__init__(self, "py_mean_squared_error")

    def delta(self, t, y, d):
        Tensor.copy(Tensor.sub(y, t), d)
        d.div_(t.shape[0])

    def value(self, t, y):
        size = t.size / t.shape[0]
        return (Tensor.sqr(Tensor.sub(t, y))).sum() / size


class MSEMetric(Metric):

    def __init__(self):
        Metric.__init__(self, "py_mean_squared_error")

    def value(self, t, y):
        size = t.size / t.shape[0]
        return (Tensor.sqr(Tensor.sub(t, y))).sum() / size


class CategoricalAccuracy(Metric):

    def __init__(self):
        Metric.__init__(self, "py_categorical_accuracy")

    def value(self, t, y):
        a = t.getdata()
        b = y.getdata()
        return (np.argmax(a, axis=-1) == np.argmax(b, axis=-1)).sum()


def test_py_metric():
    shape = [8, 10]
    a = np.random.random(shape).astype(np.float32)
    b = np.random.random(shape).astype(np.float32)
    t, y = Tensor.fromarray(a), Tensor.fromarray(b)
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
    t, y = Tensor.fromarray(a), Tensor.fromarray(b)
    z = Tensor(shape)
    exp_z = Tensor(shape)
    py_mse_loss = MSELoss()
    mse_loss = eddl.getLoss("mse")
    mse_loss.delta(t, y, exp_z)
    py_mse_loss.delta(t, y, z)
    c = np.array(z, copy=False)
    exp_c = np.array(exp_z, copy=False)
    assert np.array_equal(c, exp_c)
    v = py_mse_loss.value(t, y)
    exp_v = mse_loss.value(t, y)
    assert v == pytest.approx(exp_v)
