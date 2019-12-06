import numpy as np
import pyeddl._core.eddlT as eddlT


def test_create_getdata():
    R, C = 3, 4
    t = eddlT.create([R, C])
    assert t.shape == [R, C]
    t = eddlT.create([R, C], eddlT.DEV_CPU)
    assert t.shape == [R, C]
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    t = eddlT.create(a)
    assert t.shape == [R, C]
    b = eddlT.getdata(t)
    assert np.array_equal(b, a)
    # check automatic type conversion
    a = np.arange(R * C).reshape(R, C)
    t = eddlT.create(a)
    assert t.shape == [R, C]


def test_ones():
    shape = [2, 3]
    t = eddlT.ones(shape)
    a = np.array(t, copy=False)
    b = np.ones(shape, dtype=np.float32)
    assert np.array_equal(a, b)


def test_zeros():
    shape = [2, 3]
    t = eddlT.zeros(shape)
    a = np.array(t, copy=False)
    b = np.zeros(shape, dtype=np.float32)
    assert np.array_equal(a, b)


def test_full():
    shape = [2, 3]
    value = 42
    t = eddlT.full(shape, value)
    a = np.array(t, copy=False)
    b = np.full(shape, value, dtype=np.float32)
    assert np.array_equal(a, b)


def test_arange():
    start, stop, step = 0, 2, .33
    t = eddlT.arange(start, stop, step)
    a = np.array(t, copy=False)
    b = np.arange(start, stop, step, dtype=np.float32)
    assert np.allclose(a, b)


def test_range():
    start, stop, step = 0, 2, .33
    t = eddlT.range(start, stop, step)
    a = np.array(t, copy=False)
    b = np.arange(start, stop, step, dtype=np.float32)
    assert np.allclose(a, b)


def test_linspace():
    start, stop, num = 0, 2, 7
    t = eddlT.linspace(start, stop, num)
    a = np.array(t, copy=False)
    b = np.linspace(start, stop, num, dtype=np.float32)
    assert np.allclose(a, b)


def test_logspace():
    start, stop, num, base = 0.1, 1.0, 5, 10.0
    t = eddlT.logspace(start, stop, num, base)
    a = np.array(t, copy=False)
    b = np.logspace(start, stop, num, dtype=np.float32)
    assert np.allclose(a, b)


def test_eye():
    size = 3
    t = eddlT.eye(size)
    a = np.array(t, copy=False)
    assert np.array_equal(a.diagonal(), np.ones(size, dtype=np.float32))


def test_randn():
    shape = [2, 3]
    t = eddlT.randn(shape)
    assert t.shape == shape


def test_abs_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.abs_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.abs(a), b)


def test_acos_():
    t = eddlT.range(-1, 1, .5)
    a = np.array(t, copy=True)
    eddlT.acos_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.arccos(a), b)


def test_asin_():
    t = eddlT.range(-1, 1, .5)
    a = np.array(t, copy=True)
    eddlT.asin_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.arcsin(a), b)


def test_atan_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.atan_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.arctan(a), b)


def test_ceil_():
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.ceil_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.ceil(a), b)


def test_clamp_():
    low, high = -1, 1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.clamp_(t, low, high)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, low, high), b)


def test_clampmax_():
    high = 1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.clampmax_(t, high)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, None, high), b)


def test_clampmin_():
    low = -1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.clampmin_(t, low)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, low, None), b)


def test_cos_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.cos_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.cos(a), b)


def test_cosh_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.cosh_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.cosh(a), b)


def test_div_():
    n = 3
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.div_(t, n)
    b = np.array(t, copy=False)
    assert np.allclose(np.divide(a, n), b)


def test_exp_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.exp_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.exp(a), b)


def test_floor_():
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.floor_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.floor(a), b)


def test_log_():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.log_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.log(a), b)


def test_log2_():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.log2_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.log2(a), b)


def test_log10_():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.log10_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.log10(a), b)


def test_logn_():
    base = 3
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.logn_(t, base)
    b = np.array(t, copy=False)
    assert np.allclose(np.log(a) / np.log(base), b)


def test_max_min():
    start, stop = -4, 3
    t = eddlT.range(start, stop, 1)
    assert eddlT.max(t) == stop
    assert eddlT.min(t) == start


def test_mod_():
    n = 2
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.mod_(t, n)
    b = np.array(t, copy=False)
    assert np.allclose(np.mod(a, n), b)


def test_mult_():
    n = 2
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.mult_(t, n)
    b = np.array(t, copy=False)
    assert np.allclose(np.multiply(a, n), b)


def test_normalize_():
    m, M = -1, 1
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.normalize_(t, m, M)
    b = np.array(t, copy=False)
    r = (M - m) / (a.max() - a.min())
    assert np.allclose(r * (a - a.min()) + m, b)


def test_neg_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.neg_(t)
    b = np.array(t, copy=False)
    assert np.allclose(-a, b)


def test_reciprocal_():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.reciprocal_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.reciprocal(a), b)


def test_round_():
    t = eddlT.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=True)
    eddlT.round_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.round(a), b)


def test_rsqrt_():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.rsqrt_(t)
    b = np.array(t, copy=False)
    assert np.allclose(1 / np.sqrt(a), b)


def test_sigmoid_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sigmoid_(t)
    b = np.array(t, copy=False)
    assert np.allclose(1 / (1 + np.exp(-a)), b)


def test_sign_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sign_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.sign(a), b)


def test_sin_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sin_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.sin(a), b)


def test_sinh_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sinh_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.sinh(a), b)


def test_sqr_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sqr_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.square(a), b)


def test_sqrt_():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.sqrt_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.sqrt(a), b)


def test_sub_():
    n = 3
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sub_(t, n)
    b = np.array(t, copy=False)
    assert np.allclose(a - n, b)


def test_tan_():
    t = eddlT.range(-1, 1, .2)
    a = np.array(t, copy=True)
    eddlT.tan_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.tan(a), b)


def test_tanh_():
    t = eddlT.range(-1, 1, .2)
    a = np.array(t, copy=True)
    eddlT.tanh_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.tanh(a), b)


def test_trunc_():
    t = eddlT.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=True)
    eddlT.trunc_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.trunc(a), b)
