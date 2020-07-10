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

DEV_CPU = _core.eddl.DEV_CPU
"""\
A constant representing the CPU device
"""

DEV_GPU = _core.eddl.DEV_GPU
"""\
A constant representing the GPU device
"""


class Tensor(_core.Tensor):

    @staticmethod
    def fromarray(array, dev=DEV_CPU):
        """\
        Create a tensor from a NumPy array.

        The tensor will be initialized with values, shape, etc. from the array.

        The data type is automatically converted to float32, with one
        exception: due to the way pybind11 overloads are picked, if the input
        array is 1D, it must be of type float32. See
        http://github.com/deephealthproject/pyeddl/issues/10.

        :param array: NumPy array
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :return: Tensor
        """
        return _core.Tensor(array, dev)

    # also works if shape is a numpy array (works like fromarray)
    def __init__(self, shape, dev=DEV_CPU):
        """\
        Create an uninitialized tensor.

        :param shape: shape of the tensor to create
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :return: Tensor
        """
        _core.Tensor.__init__(self, shape, dev)

    def getdata(self):
        """\
        Get the tensor's data as a NumPy array.

        :return: a NumPy array
        """
        return _core.Tensor.getdata(self)

    # == Creation ==

    @staticmethod
    def zeros(shape, dev=DEV_CPU):
        """\
        Create a tensor of the specified shape and fill it with zeros.

        :param shape: shape of the tensor to create
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :return: Tensor
        """
        return _core.Tensor.zeros(shape, dev)

    @staticmethod
    def ones(shape, dev=DEV_CPU):
        """\
        Create a tensor of the specified shape and fill it with ones.

        :param shape: shape of the tensor to create
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :return: Tensor
        """
        return _core.Tensor.ones(shape, dev)

    @staticmethod
    def full(shape, value, dev=DEV_CPU):
        """\
        Create a tensor of the specified shape and fill it with ``value``.

        :param shape: shape of the tensor to create
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :param value: value to fill the tensor with
        :return: Tensor
        """
        return _core.Tensor.full(shape, value, dev)

    @staticmethod
    def arange(start, end, step, dev=DEV_CPU):
        """\
        Create a 1D tensor with evenly spaced values within a given interval.

        :param start: start of interval
        :param end: end of interval
        :param step: spacing between values
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :return: Tensor
        """
        return _core.Tensor.arange(start, end, step, dev)

    @staticmethod
    def range(start, end, step, dev=DEV_CPU):
        """\
        Create a 1D tensor with evenly spaced values within a given interval.

        :param start: start of interval
        :param end: end of interval
        :param step: spacing between values
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :return: Tensor
        """
        return _core.Tensor.range(start, end, step, dev)

    @staticmethod
    def linspace(start, end, steps, dev=DEV_CPU):
        """\
        Create a 1D tensor with evenly spaced values within a given interval.

        :param start: starting value
        :param end: end value
        :param steps: number of samples to generate
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :return: Tensor
        """
        return _core.Tensor.linspace(start, end, steps, dev)

    @staticmethod
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
        return _core.Tensor.logspace(start, end, steps, base, dev)

    @staticmethod
    def eye(size, offset=0, dev=DEV_CPU):
        """\
        Create a ``size x size`` tensor with ones on the diagonal and zeros
        elsewhere.

        :param size: size of the (square) tensor to create
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :return: Tensor
        """
        return _core.Tensor.eye(size, offset, dev)

    @staticmethod
    def randn(shape, dev=DEV_CPU):
        """\
        Create a tensor with normally distributed random values.

        :param shape: shape of the tensor to create
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :return: Tensor
        """
        return _core.Tensor.randn(shape, dev)

    # == Copy data ==

    def toCPU(self):
        """\
        Clone the tensor to the CPU.

        :return: None
        """
        return _core.Tensor.toCPU(self)

    def toGPU(self):
        """\
        Clone the tensor to the GPU.

        :return: None
        """
        return _core.Tensor.toGPU(self)

    def clone(self):
        """\
        Return a clone of the tensor (same device).

        :return: Tensor
        """
        return _core.Tensor.clone(self)

    def select(self, indices):
        """\
        Perform NumPy-like slicing on the tensor.

        :param indices: list of strings representing the indices to be
          selected. These indices must follow a NumPy-like syntax. For
          instance: ``["1:3", "2"]``.
        :return: Tensor
        """
        return _core.Tensor.select(self, indices)

    @staticmethod
    def copy(A, B):
        """\
        Copy data from tensor ``A`` to tensor ``B``.

        :param A: a tensor
        :param B: a tensor
        :return: None
        """
        return _core.Tensor.copy(A, B)

    # == Serialization ==

    @staticmethod
    def load(fname, format=""):
        """\
        Load a tensor from a file.

        :param fname: name of the file to load the tensor from
        :param format: file format (e.g., "bin", "jpg")
        :return: Tensor
        """
        return _core.Tensor.load(fname, format)

    def save(self, fname, format=""):
        """\
        Save the tensor to a file.

        :param fname: name of the file to save the tensor to
        :param format: file format (e.g., "bin", "jpg")
        :return: None
        """
        return _core.Tensor.save(self, fname, format)

    # == Math ==

    def abs_(self):
        """\
        Compute the element-wise absolute value of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.abs_(self)

    @staticmethod
    def abs(A):
        """\
        Compute the element-wise absolute value of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.abs(A)

    def acos_(self):
        """\
        Compute the element-wise inverse cosine of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.acos_(self)

    @staticmethod
    def acos(A):
        """\
        Compute the element-wise inverse cosine of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.acos(A)

    def add_(self, other):
        """\
        Adds ``other`` to the tensor.

        Modifies the tensor.

        :param other: a tensor or scalar
        :return: None
        """
        return _core.Tensor.add_(self, other)

    @staticmethod
    def add(A, B):
        """\
        Adds ``B`` to ``A``.

        Returns a new tensor.

        :param A: a tensor
        :param B: a tensor or scalar
        :return: Tensor
        """
        return _core.Tensor.add(A, B)

    def asin_(self):
        """\
        Compute the element-wise inverse sine of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.asin_(self)

    @staticmethod
    def asin(A):
        """\
        Compute the element-wise inverse sine of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.asin(A)

    def atan_(self):
        """\
        Compute the element-wise inverse tangent of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.atan_(self)

    @staticmethod
    def atan(A):
        """\
        Compute the element-wise inverse tangent of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.atan(A)

    def ceil_(self):
        """\
        Compute the element-wise ceiling (smallest integer i such that i >= x)
        of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.ceil_(self)

    @staticmethod
    def ceil(A):
        """\
        Compute the element-wise ceiling (smallest integer i such that i >= x)
        of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.ceil(A)

    def clamp_(self, min, max):
        """\
        Limit the tensor's values between min and max.

        Modifies the tensor.

        :param min: minimum value
        :param max: maximum value
        :return: None
        """
        return _core.Tensor.clamp_(self, min, max)

    @staticmethod
    def clamp(A, min, max):
        """\
        Limit the input tensor's values between min and max.

        Returns a new tensor.

        :param A: a tensor
        :param min: minimum value
        :param max: maximum value
        :return: Tensor
        """
        return _core.Tensor.clamp(A, min, max)

    def clampmax_(self, max):
        """\
        Limit the tensor's values to a maximum value.

        Modifies the tensor.

        :param max: maximum value
        :return: None
        """
        return _core.Tensor.clampmax_(self, max)

    @staticmethod
    def clampmax(A, max):
        """\
        Limit the input tensor's values to a maximum value.

        Returns a new tensor.

        :param A: a tensor
        :param max: maximum value
        :return: Tensor
        """
        return _core.Tensor.clampmax(A, max)

    def clampmin_(self, min):
        """\
        Limit the tensor's values to a minimum value.

        Modifies the tensor.

        :param min: minimum value
        :return: None
        """
        return _core.Tensor.clampmin_(self, min)

    @staticmethod
    def clampmin(A, min):
        """\
        Limit the input tensor's values to a minimum value.

        Returns a new tensor.

        :param A: a tensor
        :param min: minimum value
        :return: Tensor
        """
        return _core.Tensor.clampmin(A, min)

    def cos_(self):
        """\
        Compute the element-wise cosine of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.cos_(self)

    @staticmethod
    def cos(A):
        """\
        Compute the element-wise cosine of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.cos(A)

    def cosh_(self):
        """\
        Compute the element-wise hyperbolic cosine of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.cosh_(self)

    @staticmethod
    def cosh(A):
        """\
        Compute the element-wise hyperbolic cosine of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.cosh(A)

    def div_(self, other):
        """\
        Divides the tensor by ``other``.

        Modifies the tensor.

        :param other: a tensor or scalar
        :return: None
        """
        return _core.Tensor.div_(self, other)

    @staticmethod
    def div(A, B):
        """\
        Divides ``A`` by ``B``.

        Returns a new tensor.

        :param A: a tensor
        :param B: a tensor or scalar
        :return: Tensor
        """
        return _core.Tensor.div(A, B)

    def exp_(self):
        """\
        Compute the element-wise exponential of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.exp_(self)

    @staticmethod
    def exp(A):
        """\
        Compute the element-wise exponential of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.exp(A)

    def floor_(self):
        """\
        Compute the element-wise floor (largest integer i such that i <= x)
        of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.floor_(self)

    @staticmethod
    def floor(A):
        """\
        Compute the element-wise floor (largest integer i such that i <= x)
        of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.floor(A)

    def log_(self):
        """\
        Compute the element-wise natural logarithm of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.log_(self)

    @staticmethod
    def log(A):
        """\
        Compute the element-wise natural logarithm of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.log(A)

    def log2_(self):
        """\
        Compute the element-wise base-2 logarithm of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.log2_(self)

    @staticmethod
    def log2(A):
        """\
        Compute the element-wise base-2 logarithm of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.log2(A)

    def log10_(self):
        """\
        Compute the element-wise base-10 logarithm of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.log10_(self)

    @staticmethod
    def log10(A):
        """\
        Compute the element-wise base-10 logarithm of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.log10(A)

    def logn_(self, n):
        """\
        Compute the element-wise base-n logarithm of the tensor.

        Modifies the tensor.

        :param n: logarithm base
        :return: None
        """
        return _core.Tensor.logn_(self, n)

    @staticmethod
    def logn(A, n):
        """\
        Compute the element-wise base-n logarithm of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :param n: logarithm base
        :return: Tensor
        """
        return _core.Tensor.logn(A, n)

    def max(self):
        """\
        Return the maximum value of the tensor.

        :return: scalar
        """
        return _core.Tensor.max(self)

    def min(self):
        """\
        Return the minimum value of the tensor.

        :return: scalar
        """
        return _core.Tensor.min(self)

    def mod_(self, v):
        """\
        Compute the element-wise reminder of the ``A / v`` division.

        Modifies the tensor.

        :param v: a scalar
        :return: None
        """
        return _core.Tensor.mod_(self, v)

    @staticmethod
    def mod(A, v):
        """\
        Compute the element-wise reminder of the ``A / v`` division.

        Returns a new tensor.

        :param A: a tensor
        :param v: a scalar
        :return: Tensor
        """
        return _core.Tensor.mod(A, v)

    def mult_(self, other):
        """\
        Multiplies the tensor by ``other``, element-wise.

        Modifies the tensor.

        :param other: a tensor or scalar
        :return: None
        """
        return _core.Tensor.mult_(self, other)

    @staticmethod
    def mult(A, B):
        """\
        Multiplies ``A`` by ``B``, element-wise.

        Returns a new tensor.

        :param A: a tensor
        :param B: a tensor or scalar
        :return: Tensor
        """
        return _core.Tensor.mult(A, B)

    @staticmethod
    def mult2D(A, B):
        """\
        Computes the matrix product of ``A`` and ``B``.

        :param A: a tensor
        :param B: a tensor
        :return: Tensor
        """
        return _core.Tensor.mult2D(A, B)

    def neg_(self):
        """\
        Negate all elements in the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.neg_(self)

    @staticmethod
    def neg(A):
        """\
        Negate all elements in the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.neg(A)

    def normalize_(self, min, max):
        """\
        Normalize tensor values to the ``[min, max]`` range.

        ``v' = r * (v - A_min) + min; r = (max - min) / (A_max - A_min)``

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.normalize_(self, min, max)

    @staticmethod
    def normalize(A, min, max):
        """\
        Normalize input tensor values to the ``[min, max]`` range.

        ``v' = r * (v - A_min) + min; r = (max - min) / (A_max - A_min)``

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.normalize(A, min, max)

    def reciprocal_(self):
        """\
        Compute the element-wise reciprocal of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.reciprocal_(self)

    @staticmethod
    def reciprocal(A):
        """\
        Compute the element-wise reciprocal of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.reciprocal(A)

    def round_(self):
        """\
        Round tensor values to the nearest integer.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.round_(self)

    @staticmethod
    def round(A):
        """\
        Round input tensor values to the nearest integer.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.round(A)

    def rsqrt_(self):
        """\
        Compute the element-wise reciprocal of the square root of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.rsqrt_(self)

    @staticmethod
    def rsqrt(A):
        """\
        Compute the element-wise reciprocal of the square root of the input
        tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.rsqrt(A)

    def sigmoid_(self):
        """\
        Compute the element-wise sigmoid of the tensor.

        Modifies the tensor.

        :param A: a tensor
        :return: None
        """
        return _core.Tensor.sigmoid_(self)

    @staticmethod
    def sigmoid(A):
        """\
        Compute the element-wise sigmoid of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.sigmoid(A)

    def sign_(self):
        """\
        Compute the element-wise sign (-1 if x < 0, 0 if x == 0, 1 if x > 0)
        of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.sign_(self)

    @staticmethod
    def sign(A):
        """\
        Compute the element-wise sign (-1 if x < 0, 0 if x == 0, 1 if x > 0)
        of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.sign(A)

    def sin_(self):
        """\
        Compute the element-wise sine of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.sin_(self)

    @staticmethod
    def sin(A):
        """\
        Compute the element-wise sine of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.sin(A)

    def sinh_(self):
        """\
        Compute the element-wise hyperbolic sine of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.sinh_(self)

    @staticmethod
    def sinh(A):
        """\
        Compute the element-wise hyperbolic sine of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.sinh(A)

    def sqr_(self):
        """\
        Compute the element-wise square of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.sqr_(self)

    @staticmethod
    def sqr(A):
        """\
        Compute the element-wise square of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.sqr(A)

    def sqrt_(self):
        """\
        Compute the element-wise square root of the tensor.

        Modifies the input tensor.

        :return: None
        """
        return _core.Tensor.sqrt_(self)

    @staticmethod
    def sqrt(A):
        """\
        Compute the element-wise square root of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.sqrt(A)

    def sub_(self, other):
        """\
        Subtracts ``other`` from the tensor.

        Modifies the tensor.

        :param other: a tensor or scalar
        :return: None
        """
        return _core.Tensor.sub_(self, other)

    @staticmethod
    def sub(A, B):
        """\
        Subtracts ``B`` from ``A``.

        Returns a new tensor.

        :param A: a tensor
        :param B: a tensor or scalar
        :return: Tensor
        """
        return _core.Tensor.sub(A, B)

    def tan_(self):
        """\
        Compute the element-wise tangent of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.tan_(self)

    @staticmethod
    def tan(A):
        """\
        Compute the element-wise tangent of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.tan(A)

    def tanh_(self):
        """\
        Compute the element-wise hyperbolic tangent of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.tanh_(self)

    @staticmethod
    def tanh(A):
        """\
        Compute the element-wise hyperbolic tangent of the input tensor.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.tanh(A)

    def trunc_(self):
        """\
        Truncate (discard the fractional part) the tensor, element-wise.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.trunc_(self)

    @staticmethod
    def trunc(A):
        """\
        Truncate (discard the fractional part) the input tensor, element-wise.

        Returns a new tensor.

        :param A: a tensor
        :return: Tensor
        """
        return _core.Tensor.trunc(A)

    # == Other functions ==

    def fill_(self, v):
        """\
        Fill the tensor with the specified value.

        :param v: a scalar value
        :return: None
        """
        return _core.Tensor.fill_(self, v)

    def set_(self, indices, value):
        """\
        Set the tensor value to ``value`` at the specified indices.

        :param indices: a list of indices
        :param value: a scalar value
        :return: None
        """
        return _core.Tensor.set_(self, indices, value)

    def reshape_(self, new_shape):
        """\
        Change the tensor's shape.

        :param new_shape: the new shape (list of integers)
        :return: None
        """
        return _core.Tensor.reshape_(self, new_shape)

    def print(self):
        """\
        Print the tensor's values.

        :return: None
        """
        return _core.Tensor.print(self)

    def info(self):
        """\
        Print info on the tensor (shape, strides, ...).

        :return: None
        """
        return _core.Tensor.info(self)

    def getShape(self):
        """\
        Return the tensor's shape.

        :return: the tensor's shape (a list of integers)
        """
        return _core.Tensor.getShape(self)
