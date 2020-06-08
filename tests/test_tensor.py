# Copyright (c) 2019-2020, CRS4
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
    # check creation from 1D array
    a = np.array([1, 2]).astype(np.float32)
    t = Tensor(a)
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
