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


# == Creation ops ==

def create(shape, dev=None):
    # shape can also be a numpy array (when dev is None)
    if dev:
        return _eddlT.create(shape, dev)
    return _eddlT.create(shape)


def zeros(shape, dev=DEV_CPU):
    return _eddlT.zeros(shape, dev)


def ones(shape, dev=DEV_CPU):
    return _eddlT.ones(shape, dev)


def full(shape, value, dev=DEV_CPU):
    return _eddlT.full(shape, value, dev)


def arange(start, end, step, dev=DEV_CPU):
    return _eddlT.arange(start, end, step, dev)


def range(start, end, step, dev=DEV_CPU):
    return _eddlT.range(start, end, step, dev)


def linspace(start, end, steps, dev=DEV_CPU):
    return _eddlT.linspace(start, end, steps, dev)


def logspace(start, end, steps, base, dev=DEV_CPU):
    return _eddlT.logspace(start, end, steps, base, dev)


def eye(size, dev=DEV_CPU):
    return _eddlT.eye(size, dev)


def randn(shape, dev=DEV_CPU):
    return _eddlT.randn(shape, dev)


# == Copy data ==

def toCPU_(A):
    return _eddlT.toCPU_(A)


def toGPU_(A):
    return _eddlT.toGPU_(A)


def toCPU(A):
    return _eddlT.toCPU(A)


def toGPU(A):
    return _eddlT.toGPU(A)


def clone(A):
    return _eddlT.clone(A)


def select(A, i):
    return _eddlT.select(A, i)


def copyTensor(A, B):
    return _eddlT.copyTensor(A, B)


# == Core inplace ==

def fill_(A, v):
    return _eddlT.fill_(A, v)


def set_(A, indices, value):
    return _eddlT.set_(A, indices, value)


def reshape_(A, indices):
    return _eddlT.reshape_(A, indices)


# == Get data ==

def getdata(tensor):
    # returns a numpy array
    return _eddlT.getdata(tensor)


# == Other functions ==

def print(A):
    return _eddlT.print(A)


def info(A):
    return _eddlT.info(A)


def getShape(A):
    return _eddlT.getShape(A)


# == Serialization ==

def load(fname, format=""):
    return _eddlT.load(fname, format)


def save(A, fname, format=""):
    return _eddlT.save(A, fname, format)


# == Math ops ==

def abs_(A):
    return _eddlT.abs_(A)


def abs(A):
    return _eddlT.abs(A)


def acos_(A):
    return _eddlT.acos_(A)


def acos(A):
    return _eddlT.acos(A)


def add_(A, B):
    # B can be either a tensor or a float
    return _eddlT.add_(A, B)


def add(A, B):
    # B can be either a tensor or a float
    return _eddlT.add(A, B)


def asin_(A):
    return _eddlT.asin_(A)


def asin(A):
    return _eddlT.asin(A)


def atan_(A):
    return _eddlT.atan_(A)


def atan(A):
    return _eddlT.atan(A)


def ceil_(A):
    return _eddlT.ceil_(A)


def ceil(A):
    return _eddlT.ceil(A)


def clamp_(A, min, max):
    return _eddlT.clamp_(A, min, max)


def clamp(A, min, max):
    return _eddlT.clamp(A, min, max)


def clampmax_(A, max):
    return _eddlT.clampmax_(A, max)


def clampmax(A, max):
    return _eddlT.clampmax(A, max)


def clampmin_(A, min):
    return _eddlT.clampmin_(A, min)


def clampmin(A, min):
    return _eddlT.clampmin(A, min)


def cos_(A):
    return _eddlT.cos_(A)


def cos(A):
    return _eddlT.cos(A)


def cosh_(A):
    return _eddlT.cosh_(A)


def cosh(A):
    return _eddlT.cosh(A)


def div_(A, B):
    # B can be either a tensor or a float
    return _eddlT.div_(A, B)


def div(A, B):
    # B can be either a tensor or a float
    return _eddlT.div(A, B)


def exp_(A):
    return _eddlT.exp_(A)


def exp(A):
    return _eddlT.exp(A)


def floor_(A):
    return _eddlT.floor_(A)


def floor(A):
    return _eddlT.floor(A)


def inc_(A, B):
    return _eddlT.inc_(A, B)


def log_(A):
    return _eddlT.log_(A)


def log(A):
    return _eddlT.log(A)


def log2_(A):
    return _eddlT.log2_(A)


def log2(A):
    return _eddlT.log2(A)


def log10_(A):
    return _eddlT.log10_(A)


def log10(A):
    return _eddlT.log10(A)


def logn_(A, n):
    return _eddlT.logn_(A, n)


def logn(A, n):
    return _eddlT.logn(A, n)


def max(A):
    return _eddlT.max(A)


def min(A):
    return _eddlT.min(A)


def mod_(A, v):
    return _eddlT.mod_(A, v)


def mod(A, v):
    return _eddlT.mod(A, v)


def mult_(A, B):
    # B can be either a tensor or a float
    return _eddlT.mult_(A, B)


def mult(A, B):
    # B can be either a tensor or a float
    return _eddlT.mult(A, B)


def mult2D(A, B):
    return _eddlT.mult2D(A, B)


def neg_(A):
    return _eddlT.neg_(A)


def neg(A):
    return _eddlT.neg(A)


def normalize_(A, min, max):
    return _eddlT.normalize_(A, min, max)


def normalize(A, min, max):
    return _eddlT.normalize(A, min, max)


def reciprocal_(A):
    return _eddlT.reciprocal_(A)


def reciprocal(A):
    return _eddlT.reciprocal(A)


def round_(A):
    return _eddlT.round_(A)


def round(A):
    return _eddlT.round(A)


def rsqrt_(A):
    return _eddlT.rsqrt_(A)


def rsqrt(A):
    return _eddlT.rsqrt(A)


def sigmoid_(A):
    return _eddlT.sigmoid_(A)


def sigmoid(A):
    return _eddlT.sigmoid(A)


def sign_(A):
    return _eddlT.sign_(A)


def sign(A):
    return _eddlT.sign(A)


def sin_(A):
    return _eddlT.sin_(A)


def sin(A):
    return _eddlT.sin(A)


def sinh_(A):
    return _eddlT.sinh_(A)


def sinh(A):
    return _eddlT.sinh(A)


def sqr_(A):
    return _eddlT.sqr_(A)


def sqr(A):
    return _eddlT.sqr(A)


def sqrt_(A):
    return _eddlT.sqrt_(A)


def sqrt(A):
    return _eddlT.sqrt(A)


def sub_(A, B):
    # B can be either a tensor or a float
    return _eddlT.sub_(A, B)


def sub(A, B):
    # B can be either a tensor or a float
    return _eddlT.sub(A, B)


def tan_(A):
    return _eddlT.tan_(A)


def tan(A):
    return _eddlT.tan(A)


def tanh_(A):
    return _eddlT.tanh_(A)


def tanh(A):
    return _eddlT.tanh(A)


def trunc_(A):
    return _eddlT.trunc_(A)


def trunc(A):
    return _eddlT.trunc(A)


# == Reductions ==

def reduce_mean(A, axis):
    return _eddlT.reduce_mean(A, axis)


def reduce_variance(A, axis):
    return _eddlT.reduce_variance(A, axis)
