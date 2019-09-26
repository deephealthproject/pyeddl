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


def test_ones():
    shape = [2, 3]
    t = Tensor.ones(shape)
    a = np.array(t, copy=False)
    b = np.ones(shape, dtype=np.float32)
    assert np.array_equal(a, b)


def test_zeros():
    shape = [2, 3]
    t = Tensor.zeros(shape)
    a = np.array(t, copy=False)
    b = np.zeros(shape, dtype=np.float32)
    assert np.array_equal(a, b)


def test_full():
    shape = [2, 3]
    value = 42
    t = Tensor.full(shape, value)
    a = np.array(t, copy=False)
    b = np.full(shape, value, dtype=np.float32)
    assert np.array_equal(a, b)


def test_arange():
    start, stop, step = 0, 2, .33
    t = Tensor.arange(start, stop, step)
    a = np.array(t, copy=False)
    b = np.arange(start, stop, step, dtype=np.float32)
    assert np.allclose(a, b)


def test_linspace():
    start, stop, num = 0, 2, 7
    t = Tensor.linspace(start, stop, num)
    a = np.array(t, copy=False)
    b = np.linspace(start, stop, num, dtype=np.float32)
    assert np.allclose(a, b)


def test_eye():
    size = 3
    t = Tensor.eye(size)
    a = np.array(t, copy=False)
    assert np.array_equal(a.diagonal(), np.ones(size, dtype=np.float32))


def test_randn():
    shape = [2, 3]
    t = Tensor.randn(shape)
    assert t.shape == shape
