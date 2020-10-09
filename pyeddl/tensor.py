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

    def abs(self):
        """\
        Compute the element-wise absolute value of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.abs(self)

    def acos_(self):
        """\
        Compute the element-wise inverse cosine of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.acos_(self)

    def acos(self):
        """\
        Compute the element-wise inverse cosine of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.acos(self)

    def add_(self, other):
        """\
        Adds ``other`` to the tensor.

        Modifies the tensor.

        :param other: a tensor or scalar
        :return: None
        """
        return _core.Tensor.add_(self, other)

    def add(self, other):
        """\
        Adds ``other`` to the tensor.

        Returns a new tensor.

        :param other: a tensor or scalar
        :return: Tensor
        """
        return _core.Tensor.add(self, other)

    def asin_(self):
        """\
        Compute the element-wise inverse sine of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.asin_(self)

    def asin(self):
        """\
        Compute the element-wise inverse sine of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.asin(self)

    def atan_(self):
        """\
        Compute the element-wise inverse tangent of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.atan_(self)

    def atan(self):
        """\
        Compute the element-wise inverse tangent of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.atan(self)

    def ceil_(self):
        """\
        Compute the element-wise ceiling (smallest integer i such that i >= x)
        of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.ceil_(self)

    def ceil(self):
        """\
        Compute the element-wise ceiling (smallest integer i such that i >= x)
        of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.ceil(self)

    def clamp_(self, min, max):
        """\
        Limit the tensor's values between min and max.

        Modifies the tensor.

        :param min: minimum value
        :param max: maximum value
        :return: None
        """
        return _core.Tensor.clamp_(self, min, max)

    def clamp(self, min, max):
        """\
        Limit the tensor's values between min and max.

        Returns a new tensor.

        :param min: minimum value
        :param max: maximum value
        :return: Tensor
        """
        return _core.Tensor.clamp(self, min, max)

    def clampmax_(self, max):
        """\
        Limit the tensor's values to a maximum value.

        Modifies the tensor.

        :param max: maximum value
        :return: None
        """
        return _core.Tensor.clampmax_(self, max)

    def clampmax(self, max):
        """\
        Limit the tensor's values to a maximum value.

        Returns a new tensor.

        :param max: maximum value
        :return: Tensor
        """
        return _core.Tensor.clampmax(self, max)

    def clampmin_(self, min):
        """\
        Limit the tensor's values to a minimum value.

        Modifies the tensor.

        :param min: minimum value
        :return: None
        """
        return _core.Tensor.clampmin_(self, min)

    def clampmin(self, min):
        """\
        Limit the tensor's values to a minimum value.

        Returns a new tensor.

        :param min: minimum value
        :return: Tensor
        """
        return _core.Tensor.clampmin(self, min)

    def cos_(self):
        """\
        Compute the element-wise cosine of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.cos_(self)

    def cos(self):
        """\
        Compute the element-wise cosine of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.cos(self)

    def cosh_(self):
        """\
        Compute the element-wise hyperbolic cosine of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.cosh_(self)

    def cosh(self):
        """\
        Compute the element-wise hyperbolic cosine of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.cosh(self)

    def diag_(self, k=0):
        """\
        Select diagonal elements.

        Modifies the tensor (elements other than those in the selected
        diagonal are set to zero).

        :param k: offset (0 for the main diagonal, positive for the nth
          diagonal above the main one, negative for the nth diagonal below the
          main one)
        :return: None
        """
        return _core.Tensor.diag_(self, k)

    def diag(self, k=0):
        """\
        Select diagonal elements.

        Returns a new tensor which is the same as this one, except that
        elements other than those in the selected diagonal are set to zero.

        :param k: offset (0 for the main diagonal, positive for the nth
          diagonal above the main one, negative for the nth diagonal below the
          main one)
        :return: Tensor
        """
        return _core.Tensor.diag(self, k)

    def div_(self, other):
        """\
        Divides the tensor by ``other``.

        Modifies the tensor.

        :param other: a tensor or scalar
        :return: None
        """
        return _core.Tensor.div_(self, other)

    def div(self, other):
        """\
        Divides the tensor by ``other``.

        Returns a new tensor.

        :param other: a tensor or scalar
        :return: Tensor
        """
        return _core.Tensor.div(self, other)

    def exp_(self):
        """\
        Compute the element-wise exponential of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.exp_(self)

    def exp(self):
        """\
        Compute the element-wise exponential of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.exp(self)

    def floor_(self):
        """\
        Compute the element-wise floor (largest integer i such that i <= x)
        of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.floor_(self)

    def floor(self):
        """\
        Compute the element-wise floor (largest integer i such that i <= x)
        of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.floor(self)

    def log_(self):
        """\
        Compute the element-wise natural logarithm of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.log_(self)

    def log(self):
        """\
        Compute the element-wise natural logarithm of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.log(self)

    def log2_(self):
        """\
        Compute the element-wise base-2 logarithm of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.log2_(self)

    def log2(self):
        """\
        Compute the element-wise base-2 logarithm of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.log2(self)

    def log10_(self):
        """\
        Compute the element-wise base-10 logarithm of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.log10_(self)

    def log10(self):
        """\
        Compute the element-wise base-10 logarithm of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.log10(self)

    def logn_(self, n):
        """\
        Compute the element-wise base-n logarithm of the tensor.

        Modifies the tensor.

        :param n: logarithm base
        :return: None
        """
        return _core.Tensor.logn_(self, n)

    def logn(self, n):
        """\
        Compute the element-wise base-n logarithm of the tensor.

        Returns a new tensor.

        :param n: logarithm base
        :return: Tensor
        """
        return _core.Tensor.logn(self, n)

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
        Compute the element-wise reminder of the division of the tensor by
        ``v`` .

        Modifies the tensor.

        :param v: a scalar
        :return: None
        """
        return _core.Tensor.mod_(self, v)

    def mod(self, v):
        """\
        Compute the element-wise reminder of the division of the tensor by
        ``v`` .

        Returns a new tensor.

        :param v: a scalar
        :return: Tensor
        """
        return _core.Tensor.mod(self, v)

    def mult_(self, other):
        """\
        Multiplies the tensor by ``other``, element-wise.

        Modifies the tensor.

        :param other: a tensor or scalar
        :return: None
        """
        return _core.Tensor.mult_(self, other)

    def mult(self, other):
        """\
        Multiplies the tensor by ``other``, element-wise.

        Returns a new tensor.

        :param other: a tensor or scalar
        :return: Tensor
        """
        return _core.Tensor.mult(self, other)

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

    def neg(self):
        """\
        Negate all elements in the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.neg(self)

    def normalize_(self, min, max):
        """\
        Normalize tensor values to the ``[min, max]`` range.

        ``v' = r * (v - A_min) + min; r = (max - min) / (A_max - A_min)``

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.normalize_(self, min, max)

    def normalize(self, min, max):
        """\
        Normalize tensor values to the ``[min, max]`` range.

        ``v' = r * (v - A_min) + min; r = (max - min) / (A_max - A_min)``

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.normalize(self, min, max)

    def reciprocal_(self):
        """\
        Compute the element-wise reciprocal of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.reciprocal_(self)

    def reciprocal(self):
        """\
        Compute the element-wise reciprocal of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.reciprocal(self)

    def round_(self):
        """\
        Round tensor values to the nearest integer.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.round_(self)

    def round(self):
        """\
        Round tensor values to the nearest integer.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.round(self)

    def rsqrt_(self):
        """\
        Compute the element-wise reciprocal of the square root of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.rsqrt_(self)

    def rsqrt(self):
        """\
        Compute the element-wise reciprocal of the square root of the input
        tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.rsqrt(self)

    def sigmoid_(self):
        """\
        Compute the element-wise sigmoid of the tensor.

        Modifies the tensor.

        :param A: a tensor
        :return: None
        """
        return _core.Tensor.sigmoid_(self)

    def sigmoid(self):
        """\
        Compute the element-wise sigmoid of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.sigmoid(self)

    def sign_(self):
        """\
        Compute the element-wise sign (-1 if x < 0, 0 if x == 0, 1 if x > 0)
        of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.sign_(self)

    def sign(self):
        """\
        Compute the element-wise sign (-1 if x < 0, 0 if x == 0, 1 if x > 0)
        of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.sign(self)

    def sin_(self):
        """\
        Compute the element-wise sine of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.sin_(self)

    def sin(self):
        """\
        Compute the element-wise sine of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.sin(self)

    def sinh_(self):
        """\
        Compute the element-wise hyperbolic sine of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.sinh_(self)

    def sinh(self):
        """\
        Compute the element-wise hyperbolic sine of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.sinh(self)

    def sqr_(self):
        """\
        Compute the element-wise square of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.sqr_(self)

    def sqr(self):
        """\
        Compute the element-wise square of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.sqr(self)

    def sqrt_(self):
        """\
        Compute the element-wise square root of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.sqrt_(self)

    def sqrt(self):
        """\
        Compute the element-wise square root of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.sqrt(self)

    def sub_(self, other):
        """\
        Subtracts ``other`` from the tensor.

        Modifies the tensor.

        :param other: a tensor or scalar
        :return: None
        """
        return _core.Tensor.sub_(self, other)

    def sub(self, other):
        """\
        Subtracts ``other`` from the tensor.

        Returns a new tensor.

        :param other: a tensor or scalar
        :return: Tensor
        """
        return _core.Tensor.sub(self, other)

    def tan_(self):
        """\
        Compute the element-wise tangent of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.tan_(self)

    def tan(self):
        """\
        Compute the element-wise tangent of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.tan(self)

    def tanh_(self):
        """\
        Compute the element-wise hyperbolic tangent of the tensor.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.tanh_(self)

    def tanh(self):
        """\
        Compute the element-wise hyperbolic tangent of the tensor.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.tanh(self)

    def trace(self, k=0):
        """\
        Sum diagonal elements.

        :param k: offset (0 for the main diagonal, positive for the nth
          diagonal above the main one, negative for the nth diagonal below the
          main one)
        :return: float
        """
        return _core.Tensor.trace(self, k)

    def trunc_(self):
        """\
        Truncate (discard the fractional part) the tensor, element-wise.

        Modifies the tensor.

        :return: None
        """
        return _core.Tensor.trunc_(self)

    def trunc(self):
        """\
        Truncate (discard the fractional part) the tensor, element-wise.

        Returns a new tensor.

        :return: Tensor
        """
        return _core.Tensor.trunc(self)

    # == Other functions ==

    def fill_(self, v):
        """\
        Fill the tensor with the specified value.

        :param v: a scalar value
        :return: None
        """
        return _core.Tensor.fill_(self, v)

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
