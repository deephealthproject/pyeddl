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
_eddl = _core.eddl


# == MODEL METHODS ==

# = Creation =

def Model(in_, out):
    return _eddl.Model(in_, out)


def build(net, o=None, lo=None, me=None, cs=None, init_weights=True):
    # core module has multiple overloads for this:
    #  1. build(net, o=None, cs=None, init_weigths=True)
    #  2. build(net, o, lo, me, cs=None, init_weights=True)
    if (lo is None and me is not None) or (lo is not None and me is None):
        raise ValueError("lo and me must be both None or both not None")
    if lo is None:
        return _eddl.build(net, o, cs, init_weights)  # overload 1
    if o is None:
        raise ValueError("if lo and me are provided, o cannot be None")
    return _eddl.build(net, o, lo, me, cs, init_weights)  # overload 2


# = Computing services =

def toGPU(net, g=None, lsb=None, mem=None):
    # core module has multiple overloads for this:
    #  1. toGPU(net, g, lsb)
    #  2. toGPU(net, g, mem)
    #  3. toGPU(net, g, lsb, mem)
    #  4. toGPU(net, g)
    #  5. toGPU(net, mem)
    #  6. toGPU(net)
    if g is None:
        if mem is None:
            return _eddl.toGPU(net)
        return _eddl.toGPU(net, mem)
    if lsb is None:
        return _eddl.toGPU(net, g, mem)
    if mem is None:
        return _eddl.toGPU(net, g, lsb)
    return _eddl.toGPU(net, g, lsb, mem)


def toCPU(net, t=None):
    if t is None:
        return _eddl.toCPU(net)
    return _eddl.toCPU(net, t)


def CS_CPU(th=-1, mem="low_mem"):
    return _eddl.CS_CPU(th, mem)


def CS_GPU(g=[1], lsb=1, mem="low_mem"):
    return _eddl.CS_GPU(g, lsb, mem)


def CS_FGPA(f, lsb=1):
    return _eddl.CS_FGPA(f, lsb)


def CS_COMPSS(filename):
    return _eddl.CS_COMPSS(filename)


# = Info and logs =

def setlogfile(net, fname):
    return _eddl.setlogfile(net, fname)


def summary(m):
    return _eddl.summary(m)


def plot(m, fname, string="LR"):
    return _eddl.plot(m, fname, string)


# = Serialization =

def load(m, fname, format="bin"):
    return _eddl.load(m, fname, format)


def save(m, fname, format="bin"):
    return _eddl.save(m, fname, format)


# = Optimizer =

def setlr(net, p):
    return _eddl.setlr(net, p)


def adadelta(lr, rho, epsilon, weight_decay):
    return _eddl.adadelta(lr, rho, epsilon, weight_decay)


def adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.000001, weight_decay=0,
         amsgrad=False):
    return _eddl.adam(lr, beta_1, beta_2, epsilon, weight_decay, amsgrad)


def adagrad(lr, epsilon, weight_decay):
    return _eddl.adagrad(lr, epsilon, weight_decay)


def adamax(lr, beta_1, beta_2, epsilon, weight_decay):
    return _eddl.adamax(lr, beta_1, beta_2, epsilon, weight_decay)


def nadam(lr, beta_1, beta_2, epsilon, schedule_decay):
    return _eddl.nadam(lr, beta_1, beta_2, epsilon, schedule_decay)


def rmsprop(lr=0.01, rho=0.9, epsilon=0.00001, weight_decay=0.0):
    return _eddl.rmsprop(lr, rho, epsilon, weight_decay)


def sgd(lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
    return _eddl.sgd(lr, momentum, weight_decay, nesterov)


# Training and evaluation - coarse methods

def fit(m, in_, out, batch, epochs):
    return _eddl.fit(m, in_, out, batch, epochs)


def evaluate(m, in_, out):
    return _eddl.evaluate(m, in_, out)


# Training and evaluation - finer methods

def random_indices(batch_size, num_samples):
    return _eddl.random_indices(batch_size, num_samples)


def train_batch(net, in_, out, indices=None):
    if indices is None:
        return _eddl.train_batch(net, in_, out)
    return _eddl.train_batch(net, in_, out, indices)


def eval_batch(net, in_, out, indices=None):
    if indices is None:
        return _eddl.eval_batch(net, in_, out)
    return _eddl.eval_batch(net, in_, out, indices)


def next_batch(in_, out):
    return _eddl.next_batch(in_, out)


# Training and evaluation - finest methods

def set_mode(net, mode):
    return _eddl.set_mode(net, mode)


def reset_loss(m):
    return _eddl.reset_loss(m)


def forward(m, in_=None, b=None):
    # core module has multiple overloads for this:
    #  1. forward(m, in_)  where in_ is a list of layers
    #  2. forward(m, in_)  where in_ is a list of tensors
    #  3. forward(m)
    #  4. forward(m, b)
    if in_ is None:
        if b is None:
            return _eddl.forward(m)
        return _eddl.forward(m, b)
    return _eddl.forward(m, in_)


def zeroGrads(m):
    return _eddl.zeroGrads(m)


def backward(m, target=None):
    # core module has multiple overloads for this:
    #  1. backward(m, target)  where in_ is a list of layers
    #  2. backward(net)  where net is a Model
    #  3. backward(l)  where l is a NetLoss
    if target is None:
        return _eddl.backward(m)
    return _eddl.backward(m, target)


def update(m):
    return _eddl.update(m)


def print_loss(m, batch):
    return _eddl.print_loss(m, batch)


# = Model constraints =

def clamp(m, min, max):
    return _eddl.clamp(m, min, max)


# = Loss and metrics methods =

def compute_loss(L):
    return _eddl.compute_loss(L)


def compute_metric(L):
    return _eddl.compute_metric(L)


def getLoss(type_):
    return _eddl.getLoss(type_)


def newloss(f, in_, name):
    # core module has multiple overloads for this:
    #  1. newloss(f, in_, name)  where:
    #    f is a function [Layer] -> Layer
    #    in is a [Layer]
    #  2. newloss(f, in_, name)  where:
    #    f is a function Layer -> Layer
    #    in is a Layer
    return _eddl.newloss(f, in_, name)


def getMetric(type_):
    return _eddl.getMetric(type_)


# def newmetric(f, in_, name)  # TODO


def detach(l):
    # core module has multiple overloads for this:
    #  1. detach(l)  where l is a Layer and the return value is a Layer
    #  2. detach(l)  where l is a [Layer] and the return value is a [Layer]
    return _eddl.detach(l)


# == LAYERS ==

# = Core layers =

def Activation(parent, activation, params=[], name=""):
    return _eddl.Activation(parent, activation, params, name)


def Softmax(parent, name=""):
    return _eddl.Softmax(parent, name)


def Sigmoid(parent, name=""):
    return _eddl.Sigmoid(parent, name)


def HardSigmoid(parent, name=""):
    return _eddl.HardSigmoid(parent, name)


def ReLu(parent, name=""):
    return _eddl.ReLu(parent, name)


def ThresholdedReLu(parent, alpha=1.0, name=""):
    return _eddl.ThresholdedReLu(parent, alpha, name)


def LeakyReLu(parent, alpha=0.01, name=""):
    return _eddl.LeakyReLu(parent, alpha, name)


def Elu(parent, alpha=1.0, name=""):
    return _eddl.Elu(parent, alpha, name)


def Selu(parent, name=""):
    return _eddl.Selu(parent, name)


def Exponential(parent, name=""):
    return _eddl.Exponential(parent, name)


def Softplus(parent, name=""):
    return _eddl.Softplus(parent, name)


def Softsign(parent, name=""):
    return _eddl.Softsign(parent, name)


def Linear(parent, alpha=1.0, name=""):
    return _eddl.Linear(parent, alpha, name)


def Tanh(parent, name=""):
    return _eddl.Tanh(parent, name)


def Conv(parent, filters, kernel_size, strides=[1, 1], padding="same",
         use_bias=True, groups=1, dilation_rate=[1, 1], name=""):
    return _eddl.Conv(parent, filters, kernel_size, strides, padding,
                      use_bias, groups, dilation_rate, name)


def Dense(parent, ndim, use_bias=True, name=""):
    return _eddl.Dense(parent, ndim, use_bias, name)


def Dropout(parent, rate, name=""):
    return _eddl.Dropout(parent, rate, name)


def Input(shape, name=""):
    return _eddl.Input(shape, name)


def UpSampling(parent, size, interpolation="nearest", name=""):
    return _eddl.UpSampling(parent, size, interpolation, name)


def Reshape(parent, shape, name=""):
    return _eddl.Reshape(parent, shape, name)


def Flatten(parent, name=""):
    return _eddl.Flatten(parent, name)


def ConvT(parent, filters, kernel_size, output_padding, padding="same",
          dilation_rate=[1, 1], strides=[1, 1], use_bias=True, name=""):
    return _eddl.ConvT(parent, filters, kernel_size, output_padding, padding,
                       dilation_rate, strides, use_bias, name)


def Embedding(input_dim, output_dim, name=""):
    return _eddl.Embedding(input_dim, output_dim, name)


def Transpose(parent, name=""):
    return _eddl.Transpose(parent, name)


# = Transformation layers =

def Crop(parent, from_coords, to_coords, reshape=True, constant=0.0, name=""):
    return _eddl.Crop(parent, from_coords, to_coords, reshape, constant, name)


def CenteredCrop(parent, size, reshape=True, constant=0.0, name=""):
    return _eddl.CenteredCrop(parent, size, reshape, constant, name)


def CropScale(parent, from_coords, to_coords, da_mode="nearest", constant=0.0,
              name=""):
    return _eddl.CropScale(parent, from_coords, to_coords, da_mode, constant,
                           name)


def Cutout(parent, from_coords, to_coords, constant=0.0, name=""):
    return _eddl.Cutout(parent, from_coords, to_coords, constant, name)


def Flip(parent, axis=0, name=""):
    return _eddl.Flip(parent, axis, name)


def HorizontalFlip(parent, name=""):
    return _eddl.HorizontalFlip(parent, name)


def Rotate(parent, angle, offset_center=[0, 0], da_mode="original",
           constant=0.0, name=""):
    return _eddl.Rotate(parent, angle, offset_center, da_mode, constant, name)


def Scale(parent, new_shape, reshape=True, da_mode="nearest", constant=0.0,
          name=""):
    return _eddl.Scale(parent, new_shape, reshape, da_mode, constant, name)


def Shift(parent, shift, da_mode="nearest", constant=0.0, name=""):
    return _eddl.Shift(parent, shift, da_mode, constant, name)


def VerticalFlip(parent, name=""):
    return _eddl.VerticalFlip(parent, name)


# = Data augmentation layers =

def RandomCrop(parent, new_shape, name=""):
    return _eddl.RandomCrop(parent, new_shape, name)


def RandomCropScale(parent, factor, da_mode="nearest", name=""):
    return _eddl.RandomCropScale(parent, factor, da_mode, name)


def RandomCutout(parent, factor_x, factor_y, constant=0.0, name=""):
    return _eddl.RandomCutout(parent, factor_x, factor_y, constant, name)


def RandomFlip(parent, axis, name=""):
    return _eddl.RandomFlip(parent, axis, name)


def RandomHorizontalFlip(parent, name=""):
    return _eddl.RandomHorizontalFlip(parent, name)


def RandomRotation(parent, factor, offset_center=[0, 0], da_mode="original",
                   constant=0.0, name=""):
    return _eddl.RandomRotation(parent, factor, offset_center, da_mode,
                                constant, name)


def RandomScale(parent, factor, da_mode="nearest", constant=0.0, name=""):
    return _eddl.RandomScale(parent, factor, da_mode, constant, name)


def RandomShift(parent, factor_x, factor_y, da_mode="nearest", constant=0.0,
                name=""):
    return _eddl.RandomShift(parent, factor_x, factor_y, da_mode, constant,
                             name)


def RandomVerticalFlip(parent, name=""):
    return _eddl.RandomVerticalFlip(parent, name)


# = Merge layers =

def Add(layers, name=""):
    return _eddl.Add(layers, name)


def Average(layers, name=""):
    return _eddl.Average(layers, name)


def Concat(layers, axis=1, name=""):
    return _eddl.Concat(layers, axis, name)


def MatMul(layers, name=""):
    return _eddl.MatMul(layers, name)


def Maximum(layers, name=""):
    return _eddl.Maximum(layers, name)


def Minimum(layers, name=""):
    return _eddl.Minimum(layers, name)


def Subtract(layers, name=""):
    return _eddl.Subtract(layers, name)


# = Noise layers =

def GaussianNoise(parent, stddev, name=""):
    return _eddl.GaussianNoise(parent, stddev, name)


# = Normalization =

def BatchNormalization(parent, momentum=0.9, epsilon=0.001, affine=True,
                       name=""):
    return _eddl.BatchNormalization(parent, momentum, epsilon, affine, name)


def LayerNormalization(parent, momentum=0.9, epsilon=0.001, affine=True,
                       name=""):
    return _eddl.LayerNormalization(parent, momentum, epsilon, affine, name)


def GroupNormalization(parent, groups, momentum=0.9, epsilon=0.001,
                       affine=True, name=""):
    return _eddl.GroupNormalization(parent, groups, momentum, epsilon, affine,
                                    name)


def Norm(parent, epsilon=0.001, name=""):
    return _eddl.Norm(parent, epsilon, name)


def NormMax(parent, epsilon=0.001, name=""):
    return _eddl.NormMax(parent, epsilon, name)


def NormMinMax(parent, epsilon=0.001, name=""):
    return _eddl.NormMinMax(parent, epsilon, name)


# = Operator layers =

def Abs(l):
    return _eddl.Abs(l)


def Diff(l1, l2):
    # l1, l2 can be either layers or floats
    return _eddl.Diff(l1, l2)


def Div(l1, l2):
    # l1, l2 can be either layers or floats
    return _eddl.Div(l1, l2)


def Exp(l):
    return _eddl.Exp(l)


def Log(l):
    return _eddl.Log(l)


def Log2(l):
    return _eddl.Log2(l)


def Log10(l):
    return _eddl.Log10(l)


def Mult(l1, l2):
    # l1, l2 can be either layers or floats
    return _eddl.Mult(l1, l2)


def Pow(l1, l2):
    # l2 can be either a layer or a float
    return _eddl.Pow(l1, l2)


def Sqrt(l):
    return _eddl.Sqrt(l)


def Sum(l1, l2):
    # l1, l2 can be either layers or floats
    return _eddl.Sum(l1, l2)


def Select(l, indices, name=""):
    return _eddl.Select(l, indices, name)


def Permute(l, dims, name=""):
    return _eddl.Permute(l, dims, name)


# = Reduction layers =

def ReduceMean(l, axis=[0], keepdims=False):
    return _eddl.ReduceMean(l, axis, keepdims)


def ReduceVar(l, axis=[0], keepdims=False):
    return _eddl.ReduceVar(l, axis, keepdims)


def ReduceSum(l, axis=[0], keepdims=False):
    return _eddl.ReduceSum(l, axis, keepdims)


def ReduceMax(l, axis=[0], keepdims=False):
    return _eddl.ReduceMax(l, axis, keepdims)


def ReduceMin(l, axis=[0], keepdims=False):
    return _eddl.ReduceMin(l, axis, keepdims)


# = Generator layers =

def GaussGenerator(mean, stdev, size):
    return _eddl.GaussGenerator(mean, stdev, size)


def UniformGenerator(low, high, size):
    return _eddl.UniformGenerator(low, high, size)


# = Pooling layers =

def AveragePool(parent, pool_size=[2, 2], strides=[2, 2], padding="none",
                name=""):
    return _eddl.AveragePool(parent, pool_size, strides, padding, name)


def GlobalMaxPool(parent, name=""):
    return _eddl.GlobalMaxPool(parent, name)


def GlobalAveragePool(parent, name=""):
    return _eddl.GlobalAveragePool(parent, name)


def MaxPool(parent, pool_size=[2, 2], strides=[2, 2], padding="none", name=""):
    return _eddl.MaxPool(parent, pool_size, strides, padding, name)


# = Recurrent layers =

def RNN(parent, units, num_layers, use_bias=True, dropout=0.0,
        bidirectional=False, name=""):
    return _eddl.RNN(parent, units, num_layers, use_bias, dropout,
                     bidirectional, name)


def LSTM(parent, units, num_layers, use_bias=True, dropout=0.0,
         bidirectional=False, name=""):
    return _eddl.LSTM(parent, units, num_layers, use_bias, dropout,
                      bidirectional, name)


# = Layers methods =

def set_trainable(l, val):
    return _eddl.set_trainable(l, val)


def copyTensor(l1, l2):
    return _eddl.copyTensor(l1, l2)


def copyGrad(l1, l2):
    return _eddl.copyGrad(l1, l2)


def getOut(net):
    return _eddl.getOut(net)


def getTensor(l):
    return _eddl.getTensor(l)


def getGrad(l):
    return _eddl.getGrad(l)


# == INITIALIZERS ==

def GlorotNormal(l, seed=1234):
    return _eddl.GlorotNormal(l, seed)


def GlorotUniform(l, seed=1234):
    return _eddl.GlorotUniform(l, seed)


def RandomNormal(l, m=0.0, s=0.1, seed=1234):
    return _eddl.RandomNormal(l, m, s, seed)


def RandomUniform(l, min=0.0, max=0.1, seed=1234):
    return _eddl.RandomUniform(l, min, max, seed)


def Constant(l, v=0.1):
    return _eddl.Constant(l, v)


# == REGULARIZERS ==

def L2(l, l2):
    return _eddl.L2(l, l2)


def L1(l, l1):
    return _eddl.L1(l, l1)


def L1L2(l, l1, l2):
    return _eddl.L1L2(l, l1, l2)


# == DATASETS ==

def exist(name):
    return _eddl.exist(name)


def download_mnist():
    return _eddl.download_mnist()


def download_cifar10():
    return _eddl.download_cifar10()


def download_drive():
    return _eddl.download_drive()


# == ONNX ==

def save_net_to_onnx_file(net, path):
    return _eddl.save_net_to_onnx_file(net, path)


def import_net_from_onnx_file(path):
    return _eddl.import_net_from_onnx_file(path)


def serialize_net_to_onnx_string(net, gradients):
    return _eddl.serialize_net_to_onnx_string(net, gradients)


def import_net_from_onnx_string(model_string):
    return _eddl.import_net_from_onnx_string(model_string)
