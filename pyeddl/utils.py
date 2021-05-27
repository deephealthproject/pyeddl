# Copyright (c) 2019-2021 CRS4
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


class TransformationMode(_core.TransformationMode):
    """\
    Enum class which defines a tensor transformation mode.
    """
    HalfPixel = _core.TransformationMode.HalfPixel
    PytorchHalfPixel = _core.TransformationMode.PytorchHalfPixel
    AlignCorners = _core.TransformationMode.AlignCorners
    Asymmetric = _core.TransformationMode.Asymmetric
    TFCropAndResize = _core.TransformationMode.TFCropAndResize


class WrappingMode(_core.WrappingMode):
    """\
    Enum class which defines a wrapping mode for tensor transformations.
    """
    Constant = _core.WrappingMode.Constant
    Reflect = _core.WrappingMode.Reflect
    Nearest = _core.WrappingMode.Nearest
    Mirror = _core.WrappingMode.Mirror
    Wrap = _core.WrappingMode.Wrap
    Original = _core.WrappingMode.Original
