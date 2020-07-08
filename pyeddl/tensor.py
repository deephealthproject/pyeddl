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

        :param array: NumPy array
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :return: Tensor
        """
        return _core.Tensor(array, dev)

    def __init__(self, shape, dev=DEV_CPU):
        """\
        Create an uninitialized tensor.

        :param shape: shape of the tensor to create
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :return: Tensor
        """
        _core.Tensor.__init__(self, shape, dev)

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
    def eye(size, dev=DEV_CPU):
        """\
        Create a ``size x size`` tensor with ones on the diagonal and zeros
        elsewhere.

        :param size: size of the (square) tensor to create
        :param dev: device to use: :data:`DEV_CPU` or :data:`DEV_GPU`
        :return: Tensor
        """
        return _core.Tensor.eye(size, dev)

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
