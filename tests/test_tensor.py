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
from pyeddl._core import Tensor as CoreTensor
from pyeddl.tensor import Tensor as PyTensor
from pyeddl._core.eddl import DEV_CPU


# --- Creation ---

@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_create(Tensor):
    shape = [2, 3]
    t = Tensor(shape)
    assert t.shape == shape
    assert t.isCPU()
    t = Tensor(shape, DEV_CPU)
    assert t.shape == shape
    assert t.isCPU()


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_array_from_tensor(Tensor):
    shape = [2, 3]
    t = Tensor(shape)
    t.fill_(1.0)
    a = np.array(t, copy=False)
    b = np.ones(shape, dtype=np.float32)
    assert np.array_equal(a, b)
    c = np.array(t, copy=True)
    assert np.array_equal(c, b)
    d = t.getdata()
    assert np.array_equal(d, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_tensor_from_array(Tensor):
    R, C = 2, 3
    a = np.arange(6).reshape([R, C]).astype(np.float32)
    t = Tensor(a)
    assert t.shape == [R, C]
    b = np.array(t, copy=True)
    assert np.array_equal(a, b)
    # check creation from 1D array
    a = np.array([1, 2]).astype(np.float32)
    t = Tensor(a)
    b = np.array(t, copy=True)
    assert np.array_equal(a, b)
    # check automatic type conversion
    a = np.array([[1, 2], [3, 4]])
    t = Tensor(a)
    b = np.array(t, copy=True).astype(a.dtype)
    assert np.array_equal(a, b)
    # check fromarray
    if Tensor is PyTensor:
        a = np.array([1, 2]).astype(np.float32)
        t = Tensor.fromarray(a)
        b = np.array(t, copy=True)
        assert np.array_equal(a, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_tensor_array_ops(Tensor):
    a = np.arange(6).reshape([2, 3]).astype(np.float32)
    incr = 2.0
    b = a + incr
    t = Tensor(list(a.shape))
    t.fill_(incr)
    a += t
    assert np.array_equal(a, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_ones(Tensor):
    shape = [2, 3]
    t = Tensor.ones(shape)
    a = np.array(t, copy=False)
    b = np.ones(shape, dtype=np.float32)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_zeros(Tensor):
    shape = [2, 3]
    t = Tensor.zeros(shape)
    a = np.array(t, copy=False)
    b = np.zeros(shape, dtype=np.float32)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_full(Tensor):
    shape = [2, 3]
    value = 42
    t = Tensor.full(shape, value)
    a = np.array(t, copy=False)
    b = np.full(shape, value, dtype=np.float32)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_arange(Tensor):
    start, stop, step = 0, 2, .33
    t = Tensor.arange(start, stop, step)
    a = np.array(t, copy=False)
    b = np.arange(start, stop, step, dtype=np.float32)
    assert np.allclose(a, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_range(Tensor):
    start, stop, step = 0, 2, .33
    t = Tensor.range(start, stop, step)
    a = np.array(t, copy=False)
    b = np.arange(start, stop, step, dtype=np.float32)
    assert np.allclose(a, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_linspace(Tensor):
    start, stop, num = 0, 2, 7
    t = Tensor.linspace(start, stop, num)
    a = np.array(t, copy=False)
    b = np.linspace(start, stop, num, dtype=np.float32)
    assert np.allclose(a, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_logspace(Tensor):
    start, stop, num, base = 0.1, 1.0, 5, 10.0
    t = Tensor.logspace(start, stop, num, base)
    a = np.array(t, copy=False)
    b = np.logspace(start, stop, num, dtype=np.float32)
    assert np.allclose(a, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_eye(Tensor):
    size = 3
    t = Tensor.eye(size)
    a = np.array(t, copy=False)
    assert np.array_equal(a.diagonal(), np.ones(size, dtype=np.float32))
    u = Tensor.eye(size, 1)
    b = np.array(u, copy=False)
    assert(np.array_equal(b, np.eye(size, k=1)))


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_randn(Tensor):
    shape = [2, 3]
    t = Tensor.randn(shape)
    assert Tensor.getShape(t) == shape


# --- Copy ---

# toGPU fails if EDDL is compiled for GPU but no CUDA devices are
# detected (CUDA error 100) or there are driver issues (CUDA error 35).
# To conditionally skip this we probably  need some support from EDDL

# @pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
# def test_cpu_gpu(Tensor):
#     # toGPU and toCPU are no-ops if EDDL is not compiled for GPU
#     a = np.arange(6).reshape(2, 3).astype(np.float32)
#     t = Tensor(a)
#     t.toGPU()
#     t.toCPU()
#     b = np.array(t, copy=False)
#     assert np.array_equal(b, a)
#     # check getdata from GPU tensor
#     t = Tensor([2, 3], DEV_CPU)
#     t.fill_(2)
#     t.toGPU()
#     a = t.getdata()
#     assert np.alltrue(a == 2)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_clone(Tensor):
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = Tensor(a)
    u = t.clone()
    b = np.array(u, copy=False)
    assert np.array_equal(b, a)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_select(Tensor):
    R, C = 3, 4
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    t = Tensor(a)
    u = t.select(["1:3", "2"])
    b = np.array(u, copy=False)
    # numpy selection converts to row vector in this case
    assert np.array_equal(b, a[1:3, 2][:, np.newaxis])


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_copy(Tensor):
    R, C = 3, 4
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    t = Tensor(a)
    u = Tensor([R, C])
    Tensor.copy(t, u)
    b = np.array(u, copy=False)
    assert np.array_equal(b, a)


# --- Serialization ---

@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_load_save(Tensor, tmp_path):
    fn = str(tmp_path / "tensor.bin")
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = Tensor(a)
    t.save(fn, "bin")
    u = Tensor.load(fn, "bin")
    b = np.array(u, copy=False)
    assert np.array_equal(b, a)


# --- Math ---

@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_abs_(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.abs_()
    b = np.array(t, copy=False)
    assert np.allclose(np.abs(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_abs(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = Tensor.abs(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.abs(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_acos_(Tensor):
    t = Tensor.range(-1, 1, .5)
    a = np.array(t, copy=True)
    t.acos_()
    b = np.array(t, copy=False)
    assert np.allclose(np.arccos(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_acos(Tensor):
    t = Tensor.range(-1, 1, .5)
    a = np.array(t, copy=False)
    u = Tensor.acos(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.arccos(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_add_(Tensor):
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    n = 2
    # add scalar to tensor
    t = Tensor(a)
    t.add_(n)
    b = np.array(t, copy=False)
    assert np.allclose(a + n, b)
    # add tensor to tensor
    t = Tensor(a)
    c = 2 * a
    u = Tensor(c)
    t.add_(u)
    b = np.array(t, copy=False)
    assert np.allclose(b, a + c)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_add(Tensor):
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    n = 2
    # tensor = tensor + scalar
    t = Tensor(a)
    u = Tensor.add(t, n)
    b = np.array(u, copy=False)
    assert np.allclose(a + n, b)
    # tensor = tensor + tensor
    t = Tensor(a)
    u = Tensor(a + n)
    v = Tensor.add(t, u)
    b = np.array(v, copy=False)
    assert np.allclose(b, a + a + n)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_asin_(Tensor):
    t = Tensor.range(-1, 1, .5)
    a = np.array(t, copy=True)
    t.asin_()
    b = np.array(t, copy=False)
    assert np.allclose(np.arcsin(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_asin(Tensor):
    t = Tensor.range(-1, 1, .5)
    a = np.array(t, copy=False)
    u = Tensor.asin(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.arcsin(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_atan_(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.atan_()
    b = np.array(t, copy=False)
    assert np.allclose(np.arctan(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_atan(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = Tensor.atan(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.arctan(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_ceil_(Tensor):
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    t.ceil_()
    b = np.array(t, copy=False)
    assert np.allclose(np.ceil(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_ceil(Tensor):
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = Tensor.ceil(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.ceil(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_clamp_(Tensor):
    low, high = -1, 1
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    t.clamp_(low, high)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, low, high), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_clamp(Tensor):
    low, high = -1, 1
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = Tensor.clamp(t, low, high)
    b = np.array(u, copy=False)
    assert np.allclose(np.clip(a, low, high), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_clampmax_(Tensor):
    high = 1
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    t.clampmax_(high)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, None, high), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_clampmax(Tensor):
    high = 1
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = Tensor.clampmax(t, high)
    b = np.array(u, copy=False)
    assert np.allclose(np.clip(a, None, high), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_clampmin_(Tensor):
    low = -1
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    t.clampmin_(low)
    b = np.array(t, copy=False)
    assert np.allclose(np.clip(a, low, None), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_clampmin(Tensor):
    low = -1
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = Tensor.clampmin(t, low)
    b = np.array(u, copy=False)
    assert np.allclose(np.clip(a, low, None), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_cos_(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.cos_()
    b = np.array(t, copy=False)
    assert np.allclose(np.cos(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_cos(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = Tensor.cos(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.cos(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_cosh_(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.cosh_()
    b = np.array(t, copy=False)
    assert np.allclose(np.cosh(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_cosh(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = Tensor.cosh(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.cosh(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_diag_(Tensor):
    a = np.arange(16).reshape(4, 4).astype(np.float32)
    for k in range(-3, 4):
        t = Tensor(a)
        t.diag_(k=k)
        b = np.array(t, copy=False)
        exp = np.diagflat(np.diag(a, k=k), k=k)
        assert np.allclose(exp, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_diag(Tensor):
    a = np.arange(16).reshape(4, 4).astype(np.float32)
    t = Tensor(a)
    for k in range(-3, 4):
        u = t.diag(k=k)
        b = np.array(u, copy=False)
        exp = np.diagflat(np.diag(a, k=k), k=k)
        assert np.allclose(exp, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_div_(Tensor):
    a = np.arange(2, 8).reshape(2, 3).astype(np.float32)
    n = 2
    # divide scalar by tensor
    t = Tensor(a)
    t.div_(n)
    b = np.array(t, copy=False)
    assert np.allclose(a / n, b)
    # divide tensor by tensor
    t = Tensor(a)
    c = 2 * a
    u = Tensor(c)
    t.div_(u)
    b = np.array(t, copy=False)
    assert np.allclose(b, a / c)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_div(Tensor):
    a = np.arange(2, 8).reshape(2, 3).astype(np.float32)
    n = 2
    # tensor = tensor / scalar
    t = Tensor(a)
    u = Tensor.div(t, n)
    b = np.array(u, copy=False)
    assert np.allclose(a / n, b)
    # tensor = tensor / tensor
    t = Tensor(a)
    u = Tensor(a + n)
    v = Tensor.div(t, u)
    b = np.array(v, copy=False)
    assert np.allclose(b, a / (a + n))


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_exp_(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.exp_()
    b = np.array(t, copy=False)
    assert np.allclose(np.exp(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_exp(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = Tensor.exp(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.exp(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_floor_(Tensor):
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=True)
    t.floor_()
    b = np.array(t, copy=False)
    assert np.allclose(np.floor(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_floor(Tensor):
    t = Tensor.range(-2.0, 2.0, 0.25)
    a = np.array(t, copy=False)
    u = Tensor.floor(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.floor(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_log_(Tensor):
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.log_()
    b = np.array(t, copy=False)
    assert np.allclose(np.log(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_log(Tensor):
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = Tensor.log(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.log(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_log2_(Tensor):
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.log2_()
    b = np.array(t, copy=False)
    assert np.allclose(np.log2(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_log2(Tensor):
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = Tensor.log2(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.log2(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_log10_(Tensor):
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.log10_()
    b = np.array(t, copy=False)
    assert np.allclose(np.log10(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_log10(Tensor):
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = Tensor.log10(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.log10(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_logn_(Tensor):
    base = 3
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.logn_(base)
    b = np.array(t, copy=False)
    assert np.allclose(np.log(a) / np.log(base), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_logn(Tensor):
    base = 3
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = Tensor.logn(t, base)
    b = np.array(u, copy=False)
    assert np.allclose(np.log(a) / np.log(base), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_max_min(Tensor):
    start, stop = -4, 3
    t = Tensor.range(start, stop, 1)
    assert t.max() == stop
    assert t.min() == start


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_mod_(Tensor):
    n = 2
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.mod_(n)
    b = np.array(t, copy=False)
    assert np.allclose(np.mod(a, n), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_mod(Tensor):
    n = 2
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = Tensor.mod(t, n)
    b = np.array(u, copy=False)
    assert np.allclose(np.mod(a, n), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_mult_(Tensor):
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    n = 2
    # multiply scalar by tensor
    t = Tensor(a)
    t.mult_(n)
    b = np.array(t, copy=False)
    assert np.allclose(a * n, b)
    # multiply tensor by tensor
    t = Tensor(a)
    c = 2 + a
    u = Tensor(c)
    t.mult_(u)
    b = np.array(t, copy=False)
    assert np.allclose(b, a * c)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_mult(Tensor):
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    n = 2
    # tensor = tensor * scalar
    t = Tensor(a)
    u = Tensor.mult(t, n)
    b = np.array(u, copy=False)
    assert np.allclose(a * n, b)
    # tensor = tensor * tensor
    t = Tensor(a)
    u = Tensor(a + n)
    v = Tensor.mult(t, u)
    b = np.array(v, copy=False)
    assert np.allclose(b, a * (a + n))


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_mult2D(Tensor):
    a = np.arange(2, 8).reshape(2, 3).astype(np.float32)
    b = np.arange(1, 13).reshape(3, 4).astype(np.float32)
    t = Tensor(a)
    u = Tensor(b)
    v = Tensor.mult2D(t, u)
    c = np.array(v, copy=False)
    assert np.allclose(c, a @ b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_normalize_(Tensor):
    m, M = -1, 1
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.normalize_(m, M)
    b = np.array(t, copy=False)
    r = (M - m) / (a.max() - a.min())
    assert np.allclose(r * (a - a.min()) + m, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_normalize(Tensor):
    m, M = -1, 1
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = Tensor.normalize(t, m, M)
    b = np.array(u, copy=False)
    r = (M - m) / (a.max() - a.min())
    assert np.allclose(r * (a - a.min()) + m, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_neg_(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.neg_()
    b = np.array(t, copy=False)
    assert np.allclose(-a, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_neg(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = Tensor.neg(t)
    b = np.array(u, copy=False)
    assert np.allclose(-a, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_reciprocal_(Tensor):
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.reciprocal_()
    b = np.array(t, copy=False)
    assert np.allclose(np.reciprocal(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_reciprocal(Tensor):
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = Tensor.reciprocal(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.reciprocal(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_round_(Tensor):
    t = Tensor.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=True)
    t.round_()
    b = np.array(t, copy=False)
    assert np.allclose(np.round(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_round(Tensor):
    t = Tensor.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=False)
    u = Tensor.round(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.round(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_rsqrt_(Tensor):
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.rsqrt_()
    b = np.array(t, copy=False)
    assert np.allclose(1 / np.sqrt(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_rsqrt(Tensor):
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = Tensor.rsqrt(t)
    b = np.array(u, copy=False)
    assert np.allclose(1 / np.sqrt(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sigmoid_(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.sigmoid_()
    b = np.array(t, copy=False)
    assert np.allclose(1 / (1 + np.exp(-a)), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sigmoid(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = Tensor.sigmoid(t)
    b = np.array(u, copy=False)
    assert np.allclose(1 / (1 + np.exp(-a)), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sign_(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.sign_()
    b = np.array(t, copy=False)
    assert np.allclose(np.sign(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sign(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = Tensor.sign(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.sign(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sin_(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.sin_()
    b = np.array(t, copy=False)
    assert np.allclose(np.sin(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sin(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = Tensor.sin(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.sin(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sinh_(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.sinh_()
    b = np.array(t, copy=False)
    assert np.allclose(np.sinh(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sinh(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = Tensor.sinh(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.sinh(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sqr_(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=True)
    t.sqr_()
    b = np.array(t, copy=False)
    assert np.allclose(np.square(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sqr(Tensor):
    t = Tensor.range(-5, 5, 1)
    a = np.array(t, copy=False)
    u = Tensor.sqr(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.square(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sqrt_(Tensor):
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=True)
    t.sqrt_()
    b = np.array(t, copy=False)
    assert np.allclose(np.sqrt(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sqrt(Tensor):
    t = Tensor.range(1, 10, 1)
    a = np.array(t, copy=False)
    u = Tensor.sqrt(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.sqrt(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sub_(Tensor):
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    n = 2
    # subtract scalar from tensor
    t = Tensor(a)
    t.sub_(n)
    b = np.array(t, copy=False)
    assert np.allclose(a - n, b)
    # subtract tensor from tensor
    t = Tensor(a)
    c = 2 * a
    u = Tensor(c)
    t.sub_(u)
    b = np.array(t, copy=False)
    assert np.allclose(b, a - c)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_sub(Tensor):
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    n = 2
    # tensor = tensor - scalar
    t = Tensor(a)
    u = Tensor.sub(t, n)
    b = np.array(u, copy=False)
    assert np.allclose(a - n, b)
    # tensor = tensor - tensor
    t = Tensor(a)
    u = Tensor(a / n)
    v = Tensor.sub(t, u)
    b = np.array(v, copy=False)
    assert np.allclose(b, a - (a / n))


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_tan_(Tensor):
    t = Tensor.range(-1, 1, .2)
    a = np.array(t, copy=True)
    t.tan_()
    b = np.array(t, copy=False)
    assert np.allclose(np.tan(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_tan(Tensor):
    t = Tensor.range(-1, 1, .2)
    a = np.array(t, copy=False)
    u = Tensor.tan(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.tan(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_tanh_(Tensor):
    t = Tensor.range(-1, 1, .2)
    a = np.array(t, copy=True)
    t.tanh_()
    b = np.array(t, copy=False)
    assert np.allclose(np.tanh(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_tanh(Tensor):
    t = Tensor.range(-1, 1, .2)
    a = np.array(t, copy=False)
    u = Tensor.tanh(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.tanh(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_trace(Tensor):
    a = np.arange(16).reshape(4, 4).astype(np.float32)
    t = Tensor(a)
    for k in range(-3, 4):
        assert np.allclose(t.trace(k=k), np.trace(a, offset=k))


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_trunc_(Tensor):
    t = Tensor.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=True)
    t.trunc_()
    b = np.array(t, copy=False)
    assert np.allclose(np.trunc(a), b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_trunc(Tensor):
    t = Tensor.range(-2.0, 2.0, 0.4)
    a = np.array(t, copy=False)
    u = Tensor.trunc(t)
    b = np.array(u, copy=False)
    assert np.allclose(np.trunc(a), b)


# --- Other ---

@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_fill_(Tensor):
    shape = [2, 3]
    value = 42
    t = Tensor(shape)
    t.fill_(value)
    a = np.array(t, copy=False)
    b = np.full(shape, value, dtype=np.float32)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_reshape_(Tensor):
    R, C = 3, 4
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    t = Tensor(a)
    t.reshape_([C, R])
    b = np.array(t, copy=False)
    c = a.reshape(C, R)
    assert np.array_equal(b, c)


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_info_print(Tensor):
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    t = Tensor(a)
    t.info()
    t.print()


@pytest.mark.parametrize("Tensor", [CoreTensor, PyTensor])
def test_getShape(Tensor):
    shape = [3, 4]
    t = Tensor(shape)
    assert t.getShape() == shape
