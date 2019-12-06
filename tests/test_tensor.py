import numpy as np
from pyeddl._core import Tensor


def test_array_from_tensor():
    shape = [2, 3]
    t = Tensor(shape)
    assert t.shape == shape
    assert t.isCPU()
    t.fill_(1.0)
    a = np.array(t, copy=False)
    b = np.ones(shape, dtype=np.float32)
    assert np.array_equal(a, b)


def test_tensor_from_array():
    a = np.arange(6).reshape([2, 3]).astype(np.float32)
    t = Tensor(a)
    assert t.shape == list(a.shape)
    assert t.isCPU()
    b = np.array(t, copy=True)
    assert np.array_equal(a, b)


def test_tensor_array_ops():
    a = np.arange(6).reshape([2, 3]).astype(np.float32)
    incr = 2.0
    b = a + incr
    t = Tensor(list(a.shape))
    t.fill_(incr)
    a += t
    assert np.array_equal(a, b)


def test_pow_():
    n = 2
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.pow_(n)
    b = np.array(t, copy=False)
    assert np.allclose(np.power(a, n), b)


def test_remainder_():
    n = 2
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.remainder_(n)
    b = np.array(t, copy=False)
    assert np.allclose(a // n, b)


def test_sum():
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    assert np.sum(a) == t.sum()


# def test_sum_abs():
#     t = Tensor.range(1, 10, 1)
#     a = np.array(t, copy=True)
#     assert np.sum(np.abs(a)) == t.sum_abs()
