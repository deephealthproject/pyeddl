import numpy as np
import pytest
from pyeddl._core import Tensor, CustomMetric, MMeanSquaredError


def py_mse(t, y):
    aux = Tensor(t.getShape(), t.device)
    Tensor.add(1, t, -1, y, aux, 0)
    Tensor.el_mult(aux, aux, aux, 0)
    return aux.sum()


def py_mse_numpy(t, y):
    a = np.array(t, copy=False)
    b = np.array(y, copy=False)
    return np.sum(np.square(a - b))


def test_custom_metric():
    T = Tensor.ones([3, 4], 0)
    Y = Tensor([3, 4], 0)
    Y.set(0.15)
    exp_v = MMeanSquaredError().value(T, Y)
    m = CustomMetric(py_mse, "py_mean_squared_error")
    assert pytest.approx(m.value(T, Y), exp_v)
    m2 = CustomMetric(py_mse_numpy, "py_mean_squared_error")
    assert pytest.approx(m2.value(T, Y), exp_v)
