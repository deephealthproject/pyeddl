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
    start, stop = 0, 3
    t = Tensor.arange(start, stop)
    a = np.array(t, copy=False)
    b = np.arange(start, stop, dtype=np.float32)
    assert np.allclose(a, b)


def test_range():
    start, stop, step = 0, 2, .33
    t = Tensor.range(start, stop, step)
    a = np.array(t, copy=False)
    b = np.arange(start, stop, step, dtype=np.float32)
    assert np.allclose(a, b)
    start, stop = 0, 3
    t = Tensor.range(start, stop)
    a = np.array(t, copy=False)
    b = np.append(np.arange(start, stop, dtype=np.float32), stop)
    assert np.allclose(a, b)


def test_linspace():
    start, stop, num = 0, 2, 7
    t = Tensor.linspace(start, stop, num)
    a = np.array(t, copy=False)
    b = np.linspace(start, stop, num, dtype=np.float32)
    assert np.allclose(a, b)


def test_logspace():
    start, stop, num = 0.1, 1.0, 5
    t = Tensor.logspace(start, stop, num)
    a = np.array(t, copy=False)
    b = np.logspace(start, stop, num, dtype=np.float32)
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


def test_abs_():
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.abs_()
    b = np.array(t, copy=False)
    assert np.allclose(np.abs(a), b)


def test_acos_():
    t = Tensor.range(-1, 1, .5)
    a = np.array(t, copy=True)
    t.acos_()
    b = np.array(t, copy=False)
    assert np.allclose(np.arccos(a), b)


def test_asin_():
    t = Tensor.range(-1, 1, .5)
    a = np.array(t, copy=True)
    t.asin_()
    b = np.array(t, copy=False)
    assert np.allclose(np.arcsin(a), b)


def test_atan_():
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.atan_()
    b = np.array(t, copy=False)
    assert np.allclose(np.arctan(a), b)


def test_ceil_():
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    t.ceil_()
    b = np.array(t, copy=False)
    assert np.allclose(np.ceil(a), b)


def test_clamp_():
    low, high = -1, 1
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    t.clamp_(low, high)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, low, high), b)


def test_clampmax_():
    high = 1
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    t.clampmax_(high)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, None, high), b)


def test_clampmin_():
    low = -1
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    t.clampmin_(low)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, low, None), b)


def test_cos_():
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.cos_()
    b = np.array(t, copy=False)
    assert np.allclose(np.cos(a), b)


def test_cosh_():
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.cosh_()
    b = np.array(t, copy=False)
    assert np.allclose(np.cosh(a), b)


def test_div_():
    n = 3
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.div_(n)
    b = np.array(t, copy=False)
    assert np.allclose(np.divide(a, n), b)


def test_exp_():
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.exp_()
    b = np.array(t, copy=False)
    assert np.allclose(np.exp(a), b)


def test_floor_():
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    t.floor_()
    b = np.array(t, copy=False)
    assert np.allclose(np.floor(a), b)


def test_log_():
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.log_()
    b = np.array(t, copy=False)
    assert np.allclose(np.log(a), b)


def test_log2_():
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.log2_()
    b = np.array(t, copy=False)
    assert np.allclose(np.log2(a), b)


def test_log10_():
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.log10_()
    b = np.array(t, copy=False)
    assert np.allclose(np.log10(a), b)


def test_logn_():
    base = 3
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.logn_(base)
    b = np.array(t, copy=False)
    assert np.allclose(np.log(a) / np.log(base), b)


def test_max_min():
    start, stop = -4, 3
    t = Tensor.range(start, stop, 1)
    assert t.max() == stop
    assert t.min() == start


def test_mod_():
    n = 2
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.mod_(n)
    b = np.array(t, copy=False)
    assert np.allclose(np.mod(a, n), b)


def test_mult_():
    n = 2
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.mult_(n)
    b = np.array(t, copy=False)
    assert np.allclose(np.multiply(a, n), b)


def test_normalize_():
    m, M = -1, 1
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.normalize_(m, M)
    b = np.array(t, copy=False)
    r = (M - m) / (a.max() - a.min())
    assert np.allclose(r * (a - a.min()) + m, b)


def test_neg_():
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.neg_()
    b = np.array(t, copy=False)
    assert np.allclose(-a, b)


def test_pow_():
    n = 2
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.pow_(n)
    b = np.array(t, copy=False)
    assert np.allclose(np.power(a, n), b)


def test_reciprocal_():
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.reciprocal_()
    b = np.array(t, copy=False)
    assert np.allclose(np.reciprocal(a), b)


def test_remainder_():
    n = 2
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.remainder_(n)
    b = np.array(t, copy=False)
    assert np.allclose(a // n, b)


def test_round_():
    t = Tensor.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=True)
    t.round_()
    b = np.array(t, copy=False)
    assert np.allclose(np.round(a), b)


def test_rsqrt_():
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.rsqrt_()
    b = np.array(t, copy=False)
    assert np.allclose(1 / np.sqrt(a), b)


def test_sigmoid_():
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.sigmoid_()
    b = np.array(t, copy=False)
    assert np.allclose(1 / (1 + np.exp(-a)), b)


def test_sign_():
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.sign_()
    b = np.array(t, copy=False)
    assert np.allclose(np.sign(a), b)


def test_sin_():
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.sin_()
    b = np.array(t, copy=False)
    assert np.allclose(np.sin(a), b)


def test_sinh_():
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.sinh_()
    b = np.array(t, copy=False)
    assert np.allclose(np.sinh(a), b)


def test_sqr_():
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.sqr_()
    b = np.array(t, copy=False)
    assert np.allclose(np.square(a), b)


def test_sqrt_():
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.sqrt_()
    b = np.array(t, copy=False)
    assert np.allclose(np.sqrt(a), b)


def test_sub_():
    n = 3
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.sub_(n)
    b = np.array(t, copy=False)
    assert np.allclose(a - n, b)


def test_sum():
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    assert np.sum(a) == t.sum()


# def test_sum_abs():
#     t = Tensor.range(1, 10, 1)
#     a = np.array(t, copy=True)
#     assert np.sum(np.abs(a)) == t.sum_abs()


def test_tan_():
    t = Tensor.range(-1, 1, .2)
    a = np.array(t, copy=True)
    t.tan_()
    b = np.array(t, copy=False)
    assert np.allclose(np.tan(a), b)


def test_tanh_():
    t = Tensor.range(-1, 1, .2)
    a = np.array(t, copy=True)
    t.tanh_()
    b = np.array(t, copy=False)
    assert np.allclose(np.tanh(a), b)


def test_trunc_():
    t = Tensor.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=True)
    t.trunc_()
    b = np.array(t, copy=False)
    assert np.allclose(np.trunc(a), b)
