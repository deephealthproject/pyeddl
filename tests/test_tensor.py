import numpy as np
from pyeddl._core import Tensor
from pyeddl.api import DEV_CPU


def test_array_from_tensor():
    shape = [2, 3]
    t = Tensor(shape, DEV_CPU)
    assert t.shape == shape
    assert t.isCPU()
    t.set(1.0)
    a = np.array(t, copy=False)
    b = np.ones(shape, dtype=np.float32)
    assert np.array_equal(a, b)


def test_tensor_from_array():
    a = np.arange(6).reshape([2, 3]).astype(np.float32)
    t = Tensor(a, DEV_CPU)
    assert t.shape == list(a.shape)
    assert t.isCPU()
    b = np.array(t, copy=True)
    assert np.array_equal(a, b)


def test_tensor_array_ops():
    a = np.arange(6).reshape([2, 3]).astype(np.float32)
    incr = 2.0
    b = a + incr
    t = Tensor(list(a.shape), DEV_CPU)
    t.set(incr)
    a += t
    assert np.array_equal(a, b)
