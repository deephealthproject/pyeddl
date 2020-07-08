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
#     # check conversion from GPU tensor
#     t = Tensor([2, 3], DEV_CPU)
#     t.fill_(2)
#     t.toGPU()
#     a = np.array(t, copy=False)
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


# --- Other ---

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
