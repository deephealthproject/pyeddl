# Copyright (c) 2020 CRS4
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

from . import _core
_eddlT = _core.eddlT

DEV_CPU = _eddlT.DEV_CPU
"""\
A constant representing the CPU device
"""

DEV_GPU = _eddlT.DEV_GPU
"""\
A constant representing the GPU device
"""


# == Creation ops ==

def create(shape, dev=None):
    """\
    Create an uninitialized tensor of the specified shape.

    Can also be used to create a tensor from a NumPy array: ``create(array)``
    (device must be None). In this case the tensor will be initialized with
    values, shape, etc. from the array.

    :param shape: shape of the tensor to create
    :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
    :return: Tensor
    """
    # shape can also be a numpy array (when dev is None)
    if dev:
        return _eddlT.create(shape, dev)
    return _eddlT.create(shape)


def zeros(shape, dev=DEV_CPU):
    """\
    Create a tensor of the specified shape and fill it with zeros.

    :param shape: shape of the tensor to create
    :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
    :return: Tensor
    """
    return _eddlT.zeros(shape, dev)


def ones(shape, dev=DEV_CPU):
    """\
    Create a tensor of the specified shape and fill it with ones.

    :param shape: shape of the tensor to create
    :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
    :return: Tensor
    """
    return _eddlT.ones(shape, dev)


def full(shape, value, dev=DEV_CPU):
    """\
    Create a tensor of the specified shape and fill it with ``value``.

    :param shape: shape of the tensor to create
    :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
    :param value: value to fill the tensor with
    :return: Tensor
    """
    return _eddlT.full(shape, value, dev)


def arange(start, end, step, dev=DEV_CPU):
    """\
    Create a 1D tensor with evenly spaced values within a given interval.

    :param start: start of interval
    :param end: end of interval
    :param step: spacing between values
    :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
    :return: Tensor
    """
    return _eddlT.arange(start, end, step, dev)


def range(start, end, step, dev=DEV_CPU):
    """\
    Create a 1D tensor with evenly spaced values within a given interval.

    :param start: start of interval
    :param end: end of interval
    :param step: spacing between values
    :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
    :return: Tensor
    """
    return _eddlT.range(start, end, step, dev)


def linspace(start, end, steps, dev=DEV_CPU):
    """\
    Create a 1D tensor with evenly spaced values within a given interval.

    :param start: starting value
    :param end: end value
    :param steps: number of samples to generate
    :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
    :return: Tensor
    """
    return _eddlT.linspace(start, end, steps, dev)


def logspace(start, end, steps, base, dev=DEV_CPU):
    """\
    Create a 1D tensor with evenly spaced values on a log scale.

    :param start: ``base ** start`` is the starting value
    :param end: ``base ** end`` is the final value of the sequence
    :param steps: number of samples to generate
    :param base: the base of the log space
    :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
    :return: Tensor
    """
    return _eddlT.logspace(start, end, steps, base, dev)


def eye(size, dev=DEV_CPU):
    """\
    Create a ``size x size`` tensor with ones on the diagonal and zeros
    elsewhere.

    :param size: size of the (square) tensor to create
    :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
    :return: Tensor
    """
    return _eddlT.eye(size, dev)


def randn(shape, dev=DEV_CPU):
    """\
    Create a tensor with normally distributed random values.

    :param shape: shape of the tensor to create
    :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
    :return: Tensor
    """
    return _eddlT.randn(shape, dev)


# == Copy data ==

def toCPU_(A):
    """\
    Change tensor device to CPU.

    :param A: a tensor
    :return: None
    """
    return _eddlT.toCPU_(A)


def toGPU_(A):
    """\
    Change tensor device to GPU.

    :param A: a tensor
    :return: None
    """
    return _eddlT.toGPU_(A)


def toCPU(A):
    """\
    Return a clone of the input tensor, with device set to CPU.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.toCPU(A)


def toGPU(A):
    """\
    Return a clone of the input tensor, with device set to GPU.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.toGPU(A)


def clone(A):
    """\
    Return a clone of the input tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.clone(A)


def select(A, i):
    """\
    Select the i-th element (along the first axis) from a tensor and return it.

    Note that the returned tensor has the same shape as the input one.

    :param A: a tensor
    :param i: element index
    :return: Tensor
    """
    return _eddlT.select(A, i)


def copyTensor(A, B):
    """\
    Copy data from ``A`` to ``B``.

    :param A: a tensor
    :param B: a tensor
    :return: None
    """
    return _eddlT.copyTensor(A, B)


# == Core inplace ==

def fill_(A, v):
    """\
    Fill a tensor with the specified value.

    :param A: a tensor
    :param v: a scalar value
    :return: None
    """
    return _eddlT.fill_(A, v)


def set_(A, indices, value):
    """\
    Set the tensor value to ``value`` at the specified indices.

    :param A: a tensor
    :param indices: a list of indices
    :param value: a scalar value
    :return: None
    """
    return _eddlT.set_(A, indices, value)


def reshape_(A, indices):
    """\
    Change the tensor's shape.

    :param A: a tensor
    :param indices: the new shape
    :return: None
    """
    return _eddlT.reshape_(A, indices)


# == Get data ==

def getdata(tensor):
    """\
    Convert a tensor to a NumPy array.

    :param A: a tensor
    :return: a NumPy array
    """
    # returns a numpy array
    return _eddlT.getdata(tensor)


# == Other functions ==

def print(A):
    """\
    Print the tensor's values.

    :param A: a tensor
    :return: None
    """
    return _eddlT.print(A)


def info(A):
    """\
    Print info on the tensor (shape, strides, ...).

    :param A: a tensor
    :return: None
    """
    return _eddlT.info(A)


def getShape(A):
    """\
    Return the tensor's shape.

    :param A: a tensor
    :return: the tensor's shape (a list of integers)
    """
    return _eddlT.getShape(A)


# == Serialization ==

def load(fname, format=""):
    """\
    Load a tensor from a file.

    :param fname: name of the file to load the tensor from
    :param format: file format (e.g., "bin", "jpg")
    :return: Tensor
    """
    return _eddlT.load(fname, format)


def save(A, fname, format=""):
    """\
    Save a tensor to a file.

    :param A: a tensor
    :param fname: name of the file to save the tensor to
    :param format: file format (e.g., "bin", "jpg")
    :return: None
    """
    return _eddlT.save(A, fname, format)


# == Math ops ==

def abs_(A):
    """\
    Compute the element-wise absolute value of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.abs_(A)


def abs(A):
    """\
    Compute the element-wise absolute value of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.abs(A)


def acos_(A):
    """\
    Compute the element-wise inverse cosine of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.acos_(A)


def acos(A):
    """\
    Compute the element-wise inverse cosine of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.acos(A)


def add_(A, B):
    """\
    Adds ``B`` to ``A``.

    Modifies the ``A`` tensor.

    :param A: a tensor
    :param B: a tensor or scalar
    :return: None
    """
    # B can be either a tensor or a float
    return _eddlT.add_(A, B)


def add(A, B):
    """\
    Adds ``B`` to ``A``.

    Returns a new tensor.

    :param A: a tensor
    :param B: a tensor or scalar
    :return: Tensor
    """
    # B can be either a tensor or a float
    return _eddlT.add(A, B)


def asin_(A):
    """\
    Compute the element-wise inverse sine of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.asin_(A)


def asin(A):
    """\
    Compute the element-wise inverse sine of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.asin(A)


def atan_(A):
    """\
    Compute the element-wise inverse tangent of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.atan_(A)


def atan(A):
    """\
    Compute the element-wise inverse tangent of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.atan(A)


def ceil_(A):
    """\
    Compute the element-wise ceiling (smallest integer i such that i >= x)
    of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.ceil_(A)


def ceil(A):
    """\
    Compute the element-wise ceiling (smallest integer i such that i >= x)
    of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.ceil(A)


def clamp_(A, min, max):
    """\
    Limit the tensor's values between min and max.

    Modifies the input tensor.

    :param A: a tensor
    :param min: minimum value
    :param max: maximum value
    :return: None
    """
    return _eddlT.clamp_(A, min, max)


def clamp(A, min, max):
    """\
    Limit the tensor's values between min and max.

    Returns a new tensor.

    :param A: a tensor
    :param min: minimum value
    :param max: maximum value
    :return: Tensor
    """
    return _eddlT.clamp(A, min, max)


def clampmax_(A, max):
    """\
    Limit the tensor's values to a maximum value.

    Modifies the input tensor.

    :param A: a tensor
    :param max: maximum value
    :return: None
    """
    return _eddlT.clampmax_(A, max)


def clampmax(A, max):
    """\
    Limit the tensor's values to a maximum value.

    Returns a new tensor.

    :param A: a tensor
    :param max: maximum value
    :return: Tensor
    """
    return _eddlT.clampmax(A, max)


def clampmin_(A, min):
    """\
    Limit the tensor's values to a minimum value.

    Modifies the input tensor.

    :param A: a tensor
    :param min: minimum value
    :return: None
    """
    return _eddlT.clampmin_(A, min)


def clampmin(A, min):
    """\
    Limit the tensor's values to a minimum value.

    Returns a new tensor.

    :param A: a tensor
    :param min: minimum value
    :return: Tensor
    """
    return _eddlT.clampmin(A, min)


def cos_(A):
    """\
    Compute the element-wise cosine of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.cos_(A)


def cos(A):
    """\
    Compute the element-wise cosine of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.cos(A)


def cosh_(A):
    """\
    Compute the element-wise hyperbolic cosine of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.cosh_(A)


def cosh(A):
    """\
    Compute the element-wise hyperbolic cosine of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.cosh(A)


def div_(A, B):
    """\
    Divides ``A`` by ``B``.

    Modifies the ``A`` tensor.

    :param A: a tensor
    :param B: a tensor or scalar
    :return: None
    """
    # B can be either a tensor or a float
    return _eddlT.div_(A, B)


def div(A, B):
    """\
    Divides ``A`` by ``B``.

    Returns a new tensor.

    :param A: a tensor
    :param B: a tensor or scalar
    :return: Tensor
    """
    # B can be either a tensor or a float
    return _eddlT.div(A, B)


def exp_(A):
    """\
    Compute the element-wise exponential of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.exp_(A)


def exp(A):
    """\
    Compute the element-wise exponential of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.exp(A)


def floor_(A):
    """\
    Compute the element-wise floor (largest integer i such that i <= x)
    of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.floor_(A)


def floor(A):
    """\
    Compute the element-wise floor (largest integer i such that i <= x)
    of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.floor(A)


def inc_(A, B):
    """\
    Increment ``A`` by ``B``.

    Modifies the input tensor.

    :param A: a tensor
    :param B: a tensor
    :return: None
    """
    return _eddlT.inc_(A, B)


def log_(A):
    """\
    Compute the element-wise natural logarithm of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.log_(A)


def log(A):
    """\
    Compute the element-wise natural logarithm of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.log(A)


def log2_(A):
    """\
    Compute the element-wise base-2 logarithm of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.log2_(A)


def log2(A):
    """\
    Compute the element-wise base-2 logarithm of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.log2(A)


def log10_(A):
    """\
    Compute the element-wise base-10 logarithm of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.log10_(A)


def log10(A):
    """\
    Compute the element-wise base-10 logarithm of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.log10(A)


def logn_(A, n):
    """\
    Compute the element-wise base-n logarithm of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :param n: logarithm base
    :return: None
    """
    return _eddlT.logn_(A, n)


def logn(A, n):
    """\
    Compute the element-wise base-n logarithm of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :param n: logarithm base
    :return: Tensor
    """
    return _eddlT.logn(A, n)


def max(A):
    """\
    Return the maximum value of the input tensor.

    :param A: a tensor
    :return: scalar
    """
    return _eddlT.max(A)


def min(A):
    """\
    Return the minimum value of the input tensor.

    :param A: a tensor
    :return: scalar
    """
    return _eddlT.min(A)


def mod_(A, v):
    """\
    Compute the element-wise reminder of the ``A / v`` division.

    Modifies the input tensor.

    :param A: a tensor
    :param v: a scalar
    :return: None
    """
    return _eddlT.mod_(A, v)


def mod(A, v):
    """\
    Compute the element-wise reminder of the ``A / v`` division.

    Returns a new tensor.

    :param A: a tensor
    :param v: a scalar
    :return: Tensor
    """
    return _eddlT.mod(A, v)


def mult_(A, B):
    """\
    Multiplies ``A`` by ``B``, element-wise.

    Modifies the ``A`` tensor.

    :param A: a tensor
    :param B: a tensor or scalar
    :return: None
    """
    # B can be either a tensor or a float
    return _eddlT.mult_(A, B)


def mult(A, B):
    """\
    Multiplies ``A`` by ``B``, element-wise.

    Returns a new tensor.

    :param A: a tensor
    :param B: a tensor or scalar
    :return: Tensor
    """
    # B can be either a tensor or a float
    return _eddlT.mult(A, B)


def mult2D(A, B):
    """\
    Computes the matrix product of ``A`` and ``B``.

    :param A: a tensor
    :param B: a tensor
    :return: Tensor
    """
    return _eddlT.mult2D(A, B)


def neg_(A):
    """\
    Negate all elements in the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.neg_(A)


def neg(A):
    """\
    Negate all elements in the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.neg(A)


def normalize_(A, min, max):
    """\
    Normalize tensor values to the ``[min, max]`` range.

    ``v' = r * (v - A_min) + min; r = (max - min) / (A_max - A_min)``

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.normalize_(A, min, max)


def normalize(A, min, max):
    """\
    Normalize tensor values to the ``[min, max]`` range.

    ``v' = r * (v - A_min) + min; r = (max - min) / (A_max - A_min)``

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.normalize(A, min, max)


def reciprocal_(A):
    """\
    Compute the element-wise reciprocal of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.reciprocal_(A)


def reciprocal(A):
    """\
    Compute the element-wise reciprocal of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.reciprocal(A)


def round_(A):
    """\
    Round tensor values to the nearest integer.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.round_(A)


def round(A):
    """\
    Round tensor values to the nearest integer.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.round(A)


def rsqrt_(A):
    """\
    Compute the element-wise reciprocal of the square root of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.rsqrt_(A)


def rsqrt(A):
    """\
    Compute the element-wise reciprocal of the square root of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.rsqrt(A)


def sigmoid_(A):
    """\
    Compute the element-wise sigmoid of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.sigmoid_(A)


def sigmoid(A):
    """\
    Compute the element-wise sigmoid of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.sigmoid(A)


def sign_(A):
    """\
    Compute the element-wise sign (-1 if x < 0, 0 if x == 0, 1 if x > 0)
    of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.sign_(A)


def sign(A):
    """\
    Compute the element-wise sign (-1 if x < 0, 0 if x == 0, 1 if x > 0)
    of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.sign(A)


def sin_(A):
    """\
    Compute the element-wise sine of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.sin_(A)


def sin(A):
    """\
    Compute the element-wise sine of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.sin(A)


def sinh_(A):
    """\
    Compute the element-wise hyperbolic sine of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.sinh_(A)


def sinh(A):
    """\
    Compute the element-wise hyperbolic sine of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.sinh(A)


def sqr_(A):
    """\
    Compute the element-wise square of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.sqr_(A)


def sqr(A):
    """\
    Compute the element-wise square of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.sqr(A)


def sqrt_(A):
    """\
    Compute the element-wise square root of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.sqrt_(A)


def sqrt(A):
    """\
    Compute the element-wise square root of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.sqrt(A)


def sub_(A, B):
    """\
    Subtracts ``B`` from ``A``.

    Modifies the ``A`` tensor.

    :param A: a tensor
    :param B: a tensor or scalar
    :return: None
    """
    # B can be either a tensor or a float
    return _eddlT.sub_(A, B)


def sub(A, B):
    """\
    Subtracts ``B`` from ``A``.

    Returns a new tensor.

    :param A: a tensor
    :param B: a tensor or scalar
    :return: Tensor
    """
    # B can be either a tensor or a float
    return _eddlT.sub(A, B)


def tan_(A):
    """\
    Compute the element-wise tangent of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.tan_(A)


def tan(A):
    """\
    Compute the element-wise tangent of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.tan(A)


def tanh_(A):
    """\
    Compute the element-wise hyperbolic tangent of the input tensor.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.tanh_(A)


def tanh(A):
    """\
    Compute the element-wise hyperbolic tangent of the input tensor.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.tanh(A)


def trunc_(A):
    """\
    Truncate (discard the fractional part) the input tensor, element-wise.

    Modifies the input tensor.

    :param A: a tensor
    :return: None
    """
    return _eddlT.trunc_(A)


def trunc(A):
    """\
    Truncate (discard the fractional part) the input tensor, element-wise.

    Returns a new tensor.

    :param A: a tensor
    :return: Tensor
    """
    return _eddlT.trunc(A)


# == Reductions ==

def reduce_mean(A, axis):
    """\
    Compute the arithmetic mean along the specified axis.

    :param A: a tensor
    :param axis: axis (list of int) along which the mean is to be computed
    :return: Tensor
    """
    return _eddlT.reduce_mean(A, axis)


def reduce_variance(A, axis):
    """\
    Compute the variance along the specified axis.

    :param A: a tensor
    :param axis: axis (list of int) along which the variance is to be computed
    :return: Tensor
    """
    return _eddlT.reduce_variance(A, axis)
