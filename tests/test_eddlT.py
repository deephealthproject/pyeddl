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

import pytest
import numpy as np
import pyeddl._core.eddlT as eddlT_core
import pyeddl.eddlT as eddlT_py


# --- Creation ---

@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_create_getdata(eddlT):
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


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_ones(eddlT):
    shape = [2, 3]
    t = eddlT.ones(shape)
    a = np.array(t, copy=False)
    b = np.ones(shape, dtype=np.float32)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_zeros(eddlT):
    shape = [2, 3]
    t = eddlT.zeros(shape)
    a = np.array(t, copy=False)
    b = np.zeros(shape, dtype=np.float32)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_full(eddlT):
    shape = [2, 3]
    value = 42
    t = eddlT.full(shape, value)
    a = np.array(t, copy=False)
    b = np.full(shape, value, dtype=np.float32)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_arange(eddlT):
    start, stop, step = 0, 2, .33
    t = eddlT.arange(start, stop, step)
    a = np.array(t, copy=False)
    b = np.arange(start, stop, step, dtype=np.float32)
    assert np.allclose(a, b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_range(eddlT):
    start, stop, step = 0, 2, .33
    t = eddlT.range(start, stop, step)
    a = np.array(t, copy=False)
    b = np.arange(start, stop, step, dtype=np.float32)
    assert np.allclose(a, b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_linspace(eddlT):
    start, stop, num = 0, 2, 7
    t = eddlT.linspace(start, stop, num)
    a = np.array(t, copy=False)
    b = np.linspace(start, stop, num, dtype=np.float32)
    assert np.allclose(a, b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_logspace(eddlT):
    start, stop, num, base = 0.1, 1.0, 5, 10.0
    t = eddlT.logspace(start, stop, num, base)
    a = np.array(t, copy=False)
    b = np.logspace(start, stop, num, dtype=np.float32)
    assert np.allclose(a, b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_eye(eddlT):
    size = 3
    t = eddlT.eye(size)
    a = np.array(t, copy=False)
    assert np.array_equal(a.diagonal(), np.ones(size, dtype=np.float32))


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_randn(eddlT):
    shape = [2, 3]
    t = eddlT.randn(shape)
    assert eddlT.getShape(t) == shape


# --- Copy ---

@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_cpu_gpu(eddlT):
    # toGPU and toCPU are no-ops if EDDL is not compiled for GPU
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
    # check getdata from GPU tensor
    t = eddlT.create([2, 3], eddlT.DEV_CPU)
    t.fill_(2)
    eddlT.toGPU_(t)
    a = eddlT.getdata(t)
    assert np.alltrue(a == 2)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_clone(eddlT):
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    u = eddlT.clone(t)
    b = np.array(u, copy=False)
    assert np.array_equal(b, a)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_select(eddlT):
    R, C = 3, 4
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    t = eddlT.create(a)
    for i in range(R):
        u = eddlT.select(t, i)
        assert eddlT.getShape(u) == [1, C]
        b = np.array(u, copy=False)
        assert np.array_equal(b[0], a[i])


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_copyTensor(eddlT):
    R, C = 3, 4
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    t = eddlT.create(a)
    u = eddlT.create([R, C])
    eddlT.copyTensor(t, u)
    b = np.array(u, copy=False)
    assert np.array_equal(b, a)


# --- Core inplace ---

@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_fill_(eddlT):
    R, C = 2, 3
    t = eddlT.create([R, C])
    eddlT.fill_(t, 1.0)
    a = np.array(t, copy=False)
    assert np.array_equal(a, np.ones((R, C), dtype=np.float32))


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_set_(eddlT):
    R, C = 3, 4
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    n = 100.0
    t = eddlT.create(a)
    eddlT.set_(t, [1, 2], n)
    b = np.array(t, copy=False)
    a[1, 2] = n
    assert np.array_equal(b, a)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_reshape_(eddlT):
    R, C = 3, 4
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    t = eddlT.create(a)
    eddlT.reshape_(t, [C, R])
    b = np.array(t, copy=False)
    c = a.reshape(C, R)
    assert np.array_equal(b, c)


# --- Other ---

@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_info_print(eddlT):
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    eddlT.info(t)
    eddlT.print(t)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_getShape(eddlT):
    shape = [3, 4]
    t = eddlT.create(shape)
    assert eddlT.getShape(t) == shape


# --- Serialization ---

@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_load_save(eddlT, tmp_path):
    fn = str(tmp_path / "tensor.bin")
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = eddlT.create(a)
    eddlT.save(t, fn, "bin")
    u = eddlT.load(fn, "bin")
    b = np.array(u, copy=False)
    assert np.array_equal(b, a)


# --- Math ---

@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_abs_(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.abs_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.abs(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_abs(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.abs(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.abs(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_acos_(eddlT):
    t = eddlT.range(-1, 1, .5)
    a = np.array(t, copy=True)
    eddlT.acos_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.arccos(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_acos(eddlT):
    t = eddlT.range(-1, 1, .5)
    a = np.array(t, copy=False)
    u = eddlT.acos(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.arccos(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_add_(eddlT):
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


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_add(eddlT):
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


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_asin_(eddlT):
    t = eddlT.range(-1, 1, .5)
    a = np.array(t, copy=True)
    eddlT.asin_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.arcsin(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_asin(eddlT):
    t = eddlT.range(-1, 1, .5)
    a = np.array(t, copy=False)
    u = eddlT.asin(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.arcsin(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_atan_(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.atan_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.arctan(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_atan(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.atan(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.arctan(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_ceil_(eddlT):
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.ceil_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.ceil(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_ceil(eddlT):
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = eddlT.ceil(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.ceil(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_clamp_(eddlT):
    low, high = -1, 1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.clamp_(t, low, high)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, low, high), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_clamp(eddlT):
    low, high = -1, 1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = eddlT.clamp(t, low, high)
    b = np.array(u, copy=False)
    assert np.allclose(np.clip(a, low, high), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_clampmax_(eddlT):
    high = 1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.clampmax_(t, high)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, None, high), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_clampmax(eddlT):
    high = 1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = eddlT.clampmax(t, high)
    b = np.array(u, copy=False)
    assert np.allclose(np.clip(a, None, high), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_clampmin_(eddlT):
    low = -1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.clampmin_(t, low)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, low, None), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_clampmin(eddlT):
    low = -1
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = eddlT.clampmin(t, low)
    b = np.array(u, copy=False)
    assert np.allclose(np.clip(a, low, None), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_cos_(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.cos_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.cos(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_cos(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.cos(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.cos(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_cosh_(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.cosh_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.cosh(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_cosh(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.cosh(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.cosh(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_div_(eddlT):
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


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_div(eddlT):
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


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_exp_(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.exp_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.exp(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_exp(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.exp(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.exp(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_floor_(eddlT):
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    eddlT.floor_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.floor(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_floor(eddlT):
    t = eddlT.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = eddlT.floor(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.floor(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_inc_(eddlT):
    a = np.arange(2, 8).reshape(2, 3).astype(np.float32)
    n = 2
    t = eddlT.create(a)
    u = eddlT.create(a + n)
    eddlT.inc_(t, u)
    b = np.array(u, copy=False)
    assert np.allclose(b, a + a + n)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_log_(eddlT):
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.log_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.log(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_log(eddlT):
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.log(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.log(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_log2_(eddlT):
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.log2_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.log2(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_log2(eddlT):
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.log2(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.log2(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_log10_(eddlT):
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.log10_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.log10(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_log10(eddlT):
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.log10(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.log10(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_logn_(eddlT):
    base = 3
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.logn_(t, base)
    b = np.array(t, copy=False)
    assert np.allclose(np.log(a) / np.log(base), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_logn(eddlT):
    base = 3
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.logn(t, base)
    b = np.array(u, copy=False)
    assert np.allclose(np.log(a) / np.log(base), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_max_min(eddlT):
    start, stop = -4, 3
    t = eddlT.range(start, stop, 1)
    assert eddlT.max(t) == stop
    assert eddlT.min(t) == start


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_mod_(eddlT):
    n = 2
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.mod_(t, n)
    b = np.array(t, copy=False)
    assert np.allclose(np.mod(a, n), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_mod(eddlT):
    n = 2
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.mod(t, n)
    b = np.array(u, copy=False)
    assert np.allclose(np.mod(a, n), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_mult_(eddlT):
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


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_mult(eddlT):
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


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_mult2D(eddlT):
    a = np.arange(2, 8).reshape(2, 3).astype(np.float32)
    b = np.arange(1, 13).reshape(3, 4).astype(np.float32)
    t = eddlT.create(a)
    u = eddlT.create(b)
    v = eddlT.mult2D(t, u)
    c = np.array(v, copy=False)
    assert np.allclose(c, a @ b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_normalize_(eddlT):
    m, M = -1, 1
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.normalize_(t, m, M)
    b = np.array(t, copy=False)
    r = (M - m) / (a.max() - a.min())
    assert np.allclose(r * (a - a.min()) + m, b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_normalize(eddlT):
    m, M = -1, 1
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.normalize(t, m, M)
    b = np.array(u, copy=False)
    r = (M - m) / (a.max() - a.min())
    assert np.allclose(r * (a - a.min()) + m, b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_neg_(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.neg_(t)
    b = np.array(t, copy=False)
    assert np.allclose(-a, b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_neg(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.neg(t)
    b = np.array(u, copy=False)
    assert np.allclose(-a, b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_reciprocal_(eddlT):
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.reciprocal_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.reciprocal(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_reciprocal(eddlT):
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.reciprocal(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.reciprocal(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_round_(eddlT):
    t = eddlT.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=True)
    eddlT.round_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.round(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_round(eddlT):
    t = eddlT.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=False)
    u = eddlT.round(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.round(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_rsqrt_(eddlT):
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.rsqrt_(t)
    b = np.array(t, copy=False)
    assert np.allclose(1 / np.sqrt(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_rsqrt(eddlT):
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.rsqrt(t)
    b = np.array(u, copy=False)
    assert np.allclose(1 / np.sqrt(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sigmoid_(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sigmoid_(t)
    b = np.array(t, copy=False)
    assert np.allclose(1 / (1 + np.exp(-a)), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sigmoid(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.sigmoid(t)
    b = np.array(u, copy=False)
    assert np.allclose(1 / (1 + np.exp(-a)), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sign_(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sign_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.sign(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sign(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.sign(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.sign(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sin_(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sin_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.sin(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sin(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.sin(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.sin(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sinh_(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sinh_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.sinh(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sinh(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.sinh(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.sinh(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sqr_(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=True)
    eddlT.sqr_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.square(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sqr(eddlT):
    t = eddlT.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = eddlT.sqr(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.square(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sqrt_(eddlT):
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=True)
    eddlT.sqrt_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.sqrt(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sqrt(eddlT):
    t = eddlT.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = eddlT.sqrt(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.sqrt(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sub_(eddlT):
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


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_sub(eddlT):
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


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_tan_(eddlT):
    t = eddlT.range(-1, 1, .2)
    a = np.array(t, copy=True)
    eddlT.tan_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.tan(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_tan(eddlT):
    t = eddlT.range(-1, 1, .2)
    a = np.array(t, copy=False)
    u = eddlT.tan(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.tan(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_tanh_(eddlT):
    t = eddlT.range(-1, 1, .2)
    a = np.array(t, copy=True)
    eddlT.tanh_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.tanh(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_tanh(eddlT):
    t = eddlT.range(-1, 1, .2)
    a = np.array(t, copy=False)
    u = eddlT.tanh(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.tanh(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_trunc_(eddlT):
    t = eddlT.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=True)
    eddlT.trunc_(t)
    b = np.array(t, copy=False)
    assert np.allclose(np.trunc(a), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_trunc(eddlT):
    t = eddlT.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=False)
    u = eddlT.trunc(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.trunc(a), b)


# --- Reductions ---

@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_reduce_mean(eddlT):
    a = np.arange(12).reshape(3, 4).astype(np.float32)
    t = eddlT.create(a)
    for i in 0, 1:
        r = eddlT.reduce_mean(t, [i])
        b = np.array(r, copy=False)
        assert np.allclose(np.mean(a, i), b)


@pytest.mark.parametrize("eddlT", [eddlT_core, eddlT_py])
def test_reduce_variance(eddlT):
    a = np.arange(12).reshape(3, 4).astype(np.float32)
    t = eddlT.create(a)
    for i in 0, 1:
        r = eddlT.reduce_variance(t, [i])
        b = np.array(r, copy=False)
        assert np.allclose(np.var(a, i), b)
