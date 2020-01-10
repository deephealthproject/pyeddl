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
import pyeddl._core.eddlT as eddlT


# --- Creation ---

def test_create_getdata():
    R, C = 3, 4
    t = eddlT.create([R, C])
    assert eddlT.getShape(t) == [R, C]
    t = eddlT.create([R, C], eddlT.DEV_CPU)
    assert eddlT.getShape(t) == [R, C]
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    t = eddlT.create(a)
    assert eddlT.getShape(t) == [R, C]
    b = eddlT.getdata(t)
    assert np.array_equal(b, a)
    # check automatic type conversion
    a = np.arange(R * C).reshape(R, C)
    t = eddlT.create(a)
    assert eddlT.getShape(t) == [R, C]
    # check creation from 1D array
    a = np.array([1, 2]).astype(np.float32)
    t = eddlT.create(a)
    b = eddlT.getdata(t)
    assert np.array_equal(b, a)


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
    assert eddlT.getShape(t) == shape


# --- Copy ---

def test_cpu_gpu():
    # these are no-ops if EDDL is not compiled for GPU
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    eddlT.toGPU_(t)
    eddlT.toCPU_(t)
    b = np.array(t, copy=False)
    assert np.array_equal(b, a)
    u = eddlT.toGPU(t)
    t = eddlT.toCPU(u)
    b = np.array(t, copy=False)
    assert np.array_equal(b, a)


def test_clone():
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    u = eddlT.clone(t)
    b = np.array(u, copy=False)
    assert np.array_equal(b, a)


def test_select():
    R, C = 3, 4
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    t = eddlT.create(a)
    for i in range(R):
        u = eddlT.select(t, i)
        assert eddlT.getShape(u) == [1, C]
        b = np.array(u, copy=False)
        assert np.array_equal(b[0], a[i])


def test_copyTensor():
    R, C = 3, 4
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    t = eddlT.create(a)
    u = eddlT.create([R, C])
    eddlT.copyTensor(t, u)
    b = np.array(u, copy=False)
    assert np.array_equal(b, a)


# --- Core inplace ---

def test_fill_():
    R, C = 2, 3
    t = eddlT.create([R, C])
    eddlT.fill_(t, 1.0)
    a = np.array(t, copy=False)
    assert np.array_equal(a, np.ones((R, C), dtype=np.float32))


def test_set_():
    R, C = 3, 4
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    n = 100.0
    t = eddlT.create(a)
    eddlT.set_(t, [1, 2], n)
    b = np.array(t, copy=False)
    a[1, 2] = n
    assert np.array_equal(b, a)


def test_reshape_():
    R, C = 3, 4
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    t = eddlT.create(a)
    eddlT.reshape_(t, [C, R])
    b = np.array(t, copy=False)
    c = a.reshape(C, R)
    assert np.array_equal(b, c)


# --- Other ---

def test_info_print():
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    eddlT.info(t)
    eddlT.print(t)


def test_getShape():
    shape = [3, 4]
    t = eddlT.create(shape)
    assert eddlT.getShape(t) == shape


# --- Serialization ---

def test_load_save(tmp_path):
    fn = str(tmp_path / "tensor.bin")
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    eddlT.save(t, fn, "bin")
    u = eddlT.load(fn, "bin")
    b = np.array(u, copy=False)
    assert np.array_equal(b, a)


# --- Math ---

def test_abs_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.abs_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.abs(a), b)


def test_abs():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.abs(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.abs(a), b)


def test_acos_():
    t = eddlT.range(-1, 1, .5)
    a = np.array(t, copy=True)
    eddlT.acos_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.arccos(a), b)


def test_acos():
    t = eddlT.range(-1, 1, .5)
    a = np.array(t, copy=False)
    u = eddlT.acos(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.arccos(a), b)


def test_add_():
    # add scalar to tensor
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    n = 2
    eddlT.add_(t, n)
    b = np.array(t, copy=False)
    assert np.allclose(a + n, b)
    # add tensor to tensor
    t = eddlT.create(a)
    c = 2 * a
    u = eddlT.create(c)
    eddlT.add_(t, u)
    b = np.array(t, copy=False)
    assert np.allclose(b, a + c)


def test_add():
    # tensor = tensor + scalar
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    n = 2
    u = eddlT.add(t, n)
    b = np.array(u, copy=False)
    assert np.allclose(a + n, b)
    # tensor = tensor + tensor
    t = eddlT.create(a)
    u = eddlT.create(a + n)
    v = eddlT.add(t, u)
    b = np.array(v, copy=False)
    assert np.allclose(b, a + a + n)


def test_asin_():
    t = eddlT.range(-1, 1, .5)
    a = np.array(t, copy=True)
    eddlT.asin_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.arcsin(a), b)


def test_asin():
    t = eddlT.range(-1, 1, .5)
    a = np.array(t, copy=False)
    u = eddlT.asin(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.arcsin(a), b)


def test_atan_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.atan_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.arctan(a), b)


def test_atan():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.atan(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.arctan(a), b)


def test_ceil_():
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.ceil_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.ceil(a), b)


def test_ceil():
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = eddlT.ceil(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.ceil(a), b)


def test_clamp_():
    low, high = -1, 1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.clamp_(t, low, high)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, low, high), b)


def test_clamp():
    low, high = -1, 1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = eddlT.clamp(t, low, high)
    b = np.array(u, copy=False)
    assert np.allclose(np.clip(a, low, high), b)


def test_clampmax_():
    high = 1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.clampmax_(t, high)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, None, high), b)


def test_clampmax():
    high = 1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = eddlT.clampmax(t, high)
    b = np.array(u, copy=False)
    assert np.allclose(np.clip(a, None, high), b)


def test_clampmin_():
    low = -1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.clampmin_(t, low)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, low, None), b)


def test_clampmin():
    low = -1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = eddlT.clampmin(t, low)
    b = np.array(u, copy=False)
    assert np.allclose(np.clip(a, low, None), b)


def test_cos_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.cos_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.cos(a), b)


def test_cos():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.cos(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.cos(a), b)


def test_cosh_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.cosh_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.cosh(a), b)


def test_cosh():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.cosh(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.cosh(a), b)


def test_div_():
    # divide scalar by tensor
    a = np.arange(2, 8).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    n = 2
    eddlT.div_(t, n)
    b = np.array(t, copy=False)
    assert np.allclose(a / n, b)
    # divide tensor by tensor
    t = eddlT.create(a)
    c = 2 * a
    u = eddlT.create(c)
    eddlT.div_(t, u)
    b = np.array(t, copy=False)
    assert np.allclose(b, a / c)


def test_div():
    # tensor = tensor / scalar
    a = np.arange(2, 8).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    n = 2
    u = eddlT.div(t, n)
    b = np.array(u, copy=False)
    assert np.allclose(a / n, b)
    # tensor = tensor / tensor
    t = eddlT.create(a)
    u = eddlT.create(a + n)
    v = eddlT.div(t, u)
    b = np.array(v, copy=False)
    assert np.allclose(b, a / (a + n))


def test_exp_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.exp_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.exp(a), b)


def test_exp():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.exp(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.exp(a), b)


def test_floor_():
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.floor_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.floor(a), b)


def test_floor():
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = eddlT.floor(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.floor(a), b)


def test_inc_():
    a = np.arange(2, 8).reshape(2, 3).astype(np.float32)
    n = 2
    t = eddlT.create(a)
    u = eddlT.create(a + n)
    eddlT.inc_(t, u)
    b = np.array(u, copy=False)
    assert np.allclose(b, a + a + n)


def test_log_():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.log_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.log(a), b)


def test_log():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.log(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.log(a), b)


def test_log2_():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.log2_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.log2(a), b)


def test_log2():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.log2(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.log2(a), b)


def test_log10_():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.log10_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.log10(a), b)


def test_log10():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.log10(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.log10(a), b)


def test_logn_():
    base = 3
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.logn_(t, base)
    b = np.array(t, copy=False)
    assert np.allclose(np.log(a) / np.log(base), b)


def test_logn():
    base = 3
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.logn(t, base)
    b = np.array(u, copy=False)
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


def test_mod():
    n = 2
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.mod(t, n)
    b = np.array(u, copy=False)
    assert np.allclose(np.mod(a, n), b)


def test_mult_():
    # multiply scalar by tensor
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    n = 2
    eddlT.mult_(t, n)
    b = np.array(t, copy=False)
    assert np.allclose(a * n, b)
    # multiply tensor by tensor
    t = eddlT.create(a)
    c = 2 + a
    u = eddlT.create(c)
    eddlT.mult_(t, u)
    b = np.array(t, copy=False)
    assert np.allclose(b, a * c)


def test_mult():
    # tensor = tensor * scalar
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    n = 2
    u = eddlT.mult(t, n)
    b = np.array(u, copy=False)
    assert np.allclose(a * n, b)
    # tensor = tensor * tensor
    t = eddlT.create(a)
    u = eddlT.create(a + n)
    v = eddlT.mult(t, u)
    b = np.array(v, copy=False)
    assert np.allclose(b, a * (a + n))


def test_mult2D():
    a = np.arange(2, 8).reshape(2, 3).astype(np.float32)
    b = np.arange(1, 13).reshape(3, 4).astype(np.float32)
    t = eddlT.create(a)
    u = eddlT.create(b)
    v = eddlT.mult2D(t, u)
    c = np.array(v, copy=False)
    assert np.allclose(c, a @ b)


def test_normalize_():
    m, M = -1, 1
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.normalize_(t, m, M)
    b = np.array(t, copy=False)
    r = (M - m) / (a.max() - a.min())
    assert np.allclose(r * (a - a.min()) + m, b)


def test_normalize():
    m, M = -1, 1
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.normalize(t, m, M)
    b = np.array(u, copy=False)
    r = (M - m) / (a.max() - a.min())
    assert np.allclose(r * (a - a.min()) + m, b)


def test_neg_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.neg_(t)
    b = np.array(t, copy=False)
    assert np.allclose(-a, b)


def test_neg():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.neg(t)
    b = np.array(u, copy=False)
    assert np.allclose(-a, b)


def test_reciprocal_():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.reciprocal_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.reciprocal(a), b)


def test_reciprocal():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.reciprocal(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.reciprocal(a), b)


def test_round_():
    t = eddlT.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=True)
    eddlT.round_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.round(a), b)


def test_round():
    t = eddlT.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=False)
    u = eddlT.round(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.round(a), b)


def test_rsqrt_():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.rsqrt_(t)
    b = np.array(t, copy=False)
    assert np.allclose(1 / np.sqrt(a), b)


def test_rsqrt():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.rsqrt(t)
    b = np.array(u, copy=False)
    assert np.allclose(1 / np.sqrt(a), b)


def test_sigmoid_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sigmoid_(t)
    b = np.array(t, copy=False)
    assert np.allclose(1 / (1 + np.exp(-a)), b)


def test_sigmoid():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.sigmoid(t)
    b = np.array(u, copy=False)
    assert np.allclose(1 / (1 + np.exp(-a)), b)


def test_sign_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sign_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.sign(a), b)


def test_sign():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.sign(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.sign(a), b)


def test_sin_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sin_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.sin(a), b)


def test_sin():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.sin(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.sin(a), b)


def test_sinh_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sinh_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.sinh(a), b)


def test_sinh():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.sinh(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.sinh(a), b)


def test_sqr_():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sqr_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.square(a), b)


def test_sqr():
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.sqr(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.square(a), b)


def test_sqrt_():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.sqrt_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.sqrt(a), b)


def test_sqrt():
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.sqrt(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.sqrt(a), b)


def test_sub_():
    # subtract scalar from tensor
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    n = 2
    eddlT.sub_(t, n)
    b = np.array(t, copy=False)
    assert np.allclose(a - n, b)
    # subtract tensor from tensor
    t = eddlT.create(a)
    c = 2 * a
    u = eddlT.create(c)
    eddlT.sub_(t, u)
    b = np.array(t, copy=False)
    assert np.allclose(b, a - c)


def test_sub():
    # tensor = tensor - scalar
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    n = 2
    u = eddlT.sub(t, n)
    b = np.array(u, copy=False)
    assert np.allclose(a - n, b)
    # tensor = tensor - tensor
    t = eddlT.create(a)
    u = eddlT.create(a / n)
    v = eddlT.sub(t, u)
    b = np.array(v, copy=False)
    assert np.allclose(b, a - (a / n))


def test_tan_():
    t = eddlT.range(-1, 1, .2)
    a = np.array(t, copy=True)
    eddlT.tan_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.tan(a), b)


def test_tan():
    t = eddlT.range(-1, 1, .2)
    a = np.array(t, copy=False)
    u = eddlT.tan(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.tan(a), b)


def test_tanh_():
    t = eddlT.range(-1, 1, .2)
    a = np.array(t, copy=True)
    eddlT.tanh_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.tanh(a), b)


def test_tanh():
    t = eddlT.range(-1, 1, .2)
    a = np.array(t, copy=False)
    u = eddlT.tanh(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.tanh(a), b)


def test_trunc_():
    t = eddlT.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=True)
    eddlT.trunc_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.trunc(a), b)


def test_trunc():
    t = eddlT.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=False)
    u = eddlT.trunc(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.trunc(a), b)


# --- Reductions ---

def test_reduce_mean():
    a = np.arange(12).reshape(3, 4).astype(np.float32)
    t = eddlT.create(a)
    for i in 0, 1:
        r = eddlT.reduce_mean(t, [i])
        b = np.array(r, copy=False)
        assert np.allclose(np.mean(a, i), b)


def test_reduce_variance():
    a = np.arange(12).reshape(3, 4).astype(np.float32)
    t = eddlT.create(a)
    for i in 0, 1:
        r = eddlT.reduce_variance(t, [i])
        b = np.array(r, copy=False)
        assert np.allclose(np.var(a, i), b)
