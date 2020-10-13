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

from . import _core
_eddl = _core.eddl


# == MODEL METHODS ==

# = Creation =

def Model(in_, out=None):
    r"""\
    Create a model (Net).

    With one argument, create from a list of models. With two arguments,
    create from a list of input layers and a list of output layers.

    :param in\_: if this is the only parameter, it must be a list of models.
      If ``out`` is specified, this must be a list of input layers.
    :param out: list of output layers
    :return: model instance
    """
    if out is None:
        return _eddl.Model(in_)
    return _eddl.Model(in_, out)


def setName(m, name):
    return _eddl.setName(m, name)


def getLayer(net, in_):
    return _eddl.getLayer(net, in_)


def build(net, o=None, lo=None, me=None, cs=None, init_weights=True):
    """\
    Tell the model which optimizer, losses, metrics and computing service to
    use.

    :param net: model
    :param o: optimizer
    :param lo: list of losses
    :param me: list of metrics
    :param cs: computing service
    :return: None
    """
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
    """\
    Assign model operations to the GPU.

    :param net: model
    :param g: list of gpu ids to allocate
    :param lsb: number of batches to sync model weights
    :return: None
    """
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
    """\
    Assign model operations to the CPU.

    :param net: model
    :param t: number of CPU threads
    :return: None
    """
    if t is None:
        return _eddl.toCPU(net)
    return _eddl.toCPU(net, t)


def CS_CPU(th=-1, mem="full_mem"):
    """\
    Create a computing service that executes the code in the CPU.

    :param th: number of threads to use (-1 = all available threads)
    :param mem: memory consumption of the model: "full_mem", "mid_mem" or
      "low_mem"
    :return: computing service
    """
    return _eddl.CS_CPU(th, mem)


def CS_GPU(g=[1], lsb=1, mem="full_mem"):
    """\
    Create a computing service that executes the code in the GPU.

    :param g: list of integers to set which GPUs will be used (1=on, 0=off)
    :param lsb: (multi-gpu setting) number of batches to run before
      synchronizing the weights of the different GPUs
    :param mem: memory consumption of the model: "full_mem", "mid_mem" or
      "low_mem"
    :return: computing service
    """
    return _eddl.CS_GPU(g, lsb, mem)


def CS_FPGA(f, lsb=1):
    """\
    Create a computing service that executes the code in the FPGA.

    :param f: list of integers to set which FPGAs will be used (1=on, 0=off)
    :param lsb: (multi-fpga setting) number of batches to run before
      synchronizing the weights of the different FPGAs
    :return: computing service
    """
    return _eddl.CS_FPGA(f, lsb)


def CS_COMPSS(filename):
    """\
    Create a computing service that executes the code in the COMPSs framework.

    :param filename: file with the setup specification
    :return: computing service
    """
    return _eddl.CS_COMPSS(filename)


# = Info and logs =

def setlogfile(net, fname):
    """\
    Save the training outputs of a model to a file.

    :param net: model
    :param fname: name of the log file
    :return: None
    """
    return _eddl.setlogfile(net, fname)


def summary(m):
    """\
    Print a summary representation of the model.

    :param m: model
    :return: None
    """
    return _eddl.summary(m)


def plot(m, fname, string="LR"):
    """\
    Plot a representation of the model.

    :param m: model to plot
    :param fname: name of the file where the plot should be saved
    :return: None
    """
    return _eddl.plot(m, fname, string)


# = Serialization =

def load(m, fname, format="bin"):
    """\
    Load weights to reinstantiate the model.

    :param m: model
    :param fname: name of the file containing the model weights
    :return: None
    """
    return _eddl.load(m, fname, format)


def save(m, fname, format="bin"):
    """\
    Save model weights to a file.

    :param m: model
    :param fname: name of the file where model weights should be saved
    :return: None
    """
    return _eddl.save(m, fname, format)


# = Optimizer =

def setlr(net, p):
    """\
    Change the learning rate and hyperparameters of the model optimizer.

    :param net: model
    :param p: list with the learning rate and hyperparameters of the model
    :return: None
    """
    return _eddl.setlr(net, p)


def adadelta(lr, rho, epsilon, weight_decay):
    """\
    Adadelta optimizer.

    Adadelta is a more robust extension of Adagrad that adapts learning rates
    based on a moving window of gradient updates, instead of accumulating all
    past gradients. This way, Adadelta continues learning even when many
    updates have been done. See: https://arxiv.org/abs/1212.5701.

    :param lr: learning rate
    :param rho: smoothing constant
    :param epsilon: term added to the denominator to improve numerical
      stability
    :param weight_decay: weight decay (L2 penalty)
    :return: Adadelta optimizer
    """
    return _eddl.adadelta(lr, rho, epsilon, weight_decay)


def adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.000001, weight_decay=0,
         amsgrad=False):
    """\
    Adam optimizer.

    Default parameters follow those provided in the original paper
    (see: https://arxiv.org/abs/1412.6980v8).

    :param lr: learning rate
    :param beta_1: coefficient for computing running averages of gradient
      and its square
    :param beta_2: coefficient for computing running averages of gradient
      and its square
    :param epsilon: term added to the denominator to improve numerical
      stability
    :param weight_decay: weight decay (L2 penalty)
    :param amsgrad: whether to apply the AMSGrad variant of this algorithm
      from the paper "On the Convergence of Adam and Beyond"
    :return: Adam optimizer
    """
    return _eddl.adam(lr, beta_1, beta_2, epsilon, weight_decay, amsgrad)


def adagrad(lr, epsilon, weight_decay):
    """\
    Adagrad optimizer.

    Adagrad is an optimizer with parameter-specific learning rates, which are
    adapted relative to how frequently a parameter gets updated during
    training. The more updates a parameter receives, the smaller the learning
    rate. See: http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

    :param lr: learning rate
    :param epsilon: term added to the denominator to improve numerical
      stability
    :param weight_decay: weight decay (L2 penalty)
    :return: Adagrad optimizer
    """
    return _eddl.adagrad(lr, epsilon, weight_decay)


def adamax(lr, beta_1, beta_2, epsilon, weight_decay):
    """\
    Adamax optimizer.

    A variant of Adam based on the infinity norm.

    :param lr: learning rate
    :param beta_1: coefficient for computing running averages of gradient
      and its square
    :param beta_2: coefficient for computing running averages of gradient
      and its square
    :param epsilon: term added to the denominator to improve numerical
      stability
    :param weight_decay: weight decay (L2 penalty)
    :return: Adamax optimizer
    """
    return _eddl.adamax(lr, beta_1, beta_2, epsilon, weight_decay)


def nadam(lr, beta_1, beta_2, epsilon, schedule_decay):
    """\
    Nadam optimizer.

    :param lr: learning rate
    :param beta_1: coefficients for computing running averages of gradient
      and its square
    :param beta_2: coefficients for computing running averages of gradient
      and its square
    :param epsilon: term added to the denominator to improve numerical
      stability
    :param schedule_decay: weight decay (L2 penalty)
    :return: Nadam optimizer
    """
    return _eddl.nadam(lr, beta_1, beta_2, epsilon, schedule_decay)


def rmsprop(lr=0.01, rho=0.9, epsilon=0.00001, weight_decay=0.0):
    """\
    RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer at their
    default values (except for the learning rate, which can be freely tuned).
    See:
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    :param lr: learning rate
    :param rho: smoothing constant
    :param epsilon: term added to the denominator to improve numerical
      stability
    :param weight_decay: weight decay (L2 penalty)
    :return: RMSProp optimizer
    """
    return _eddl.rmsprop(lr, rho, epsilon, weight_decay)


def sgd(lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
    """\
    Stochastic gradient descent optimizer.

    Includes support for momentum, learning rate decay, and Nesterov momentum.

    :param lr: learning rate
    :param momentum: momentum factor
    :param weight_decay: value to apply to the activation function
    :param nesterov: whether to apply Nesterov momentum
    :return: SGD optimizer
    """
    return _eddl.sgd(lr, momentum, weight_decay, nesterov)


# Training and evaluation - coarse methods

def fit(m, in_, out, batch, epochs):
    """\
    Train the model for a fixed number of epochs (iterations on a dataset).

    :param m: model to train
    :param in_: input data (features)
    :param out: output data (labels)
    :param batch: number of samples per gradient update
    :param epochs: number of training epochs. An epoch is an iteration over
      the entire data provided
    :return: None
    """
    return _eddl.fit(m, in_, out, batch, epochs)


def evaluate(m, in_, out):
    """\
    Compute the loss and metric values for the model in test mode.

    :param m: model to train
    :param in_: input data (features)
    :param out: output data (labels)
    :return: None
    """
    return _eddl.evaluate(m, in_, out)


def predict(m, in_):
    r"""\
    Perform a prediction with input data

    :param m: model
    :param in\_: input data (features)
    :return: output tensors
    """
    return _eddl.predict(m, in_)


# Training and evaluation - finer methods

def random_indices(batch_size, num_samples):
    """\
    Generate a random sequence of indices for a batch.

    :param batch_size: length of the random sequence to generate
    :param num_samples: number of samples available, i.e., maximum value to
      include in the random sequence + 1
    :return: list of integers
    """
    return _eddl.random_indices(batch_size, num_samples)


def train_batch(net, in_, out, indices=None):
    """\
    Train the model using the samples of the input list that are on the
    selected indices list.

    :param net: model to train
    :param in_: list of samples
    :param out: list of labels or expected output
    :param indices: list of indices of the samples to train
    :return: None
    """
    if indices is None:
        return _eddl.train_batch(net, in_, out)
    return _eddl.train_batch(net, in_, out, indices)


def eval_batch(net, in_, out, indices=None):
    """\
    Evaluate the model using the samples of the input list that are on the
    selected indices list.

    :param net: model to evaluate
    :param in_: list of samples
    :param out: list of labels or expected output
    :param indices: list of indices of the samples to train
    :return: None
    """
    if indices is None:
        return _eddl.eval_batch(net, in_, out)
    return _eddl.eval_batch(net, in_, out, indices)


def next_batch(in_, out):
    """\
    Load the next batch of random samples from the input list to the
    output list.

    :param in_: list from where the samples of the next batch should be
      chosen from
    :param out: list where the samples of the next batch should be stored
    :return: None
    """
    return _eddl.next_batch(in_, out)


# Training and evaluation - finest methods

def set_mode(net, mode):
    """\
    Set model mode.

    :param net: model
    :param mode: 1 for training, 0 for test
    :return: None
    """
    return _eddl.set_mode(net, mode)


def reset_loss(m):
    """\
    Reset model loss.

    :param m: model
    :return: None
    """
    return _eddl.reset_loss(m)


def forward(m, in_=None, b=None):
    """\
    Compute the gradient of the model through the forward graph

    :param m: model
    :param in_: list of layers or tensors
    :param b: batch size to resize the model to
    :return: list of layers
    """
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
    """\
    Set model gradients to zero.

    :param net: model
    :return: None
    """
    return _eddl.zeroGrads(m)


def backward(m, target=None):
    """\
    Calculate the gradient by passing its argument (1x1 unit tensor by
    default) through the backward graph.

    :param m: model or loss (if it's a loss then target must not be provided)
    :param target: list of tensors
    :return: None
    """
    # core module has multiple overloads for this:
    #  1. backward(m, target)  where m is a Model and target a list of Tensor
    #  2. backward(net)  where net is a Model
    #  3. backward(l)  where l is a NetLoss
    if target is None:
        return _eddl.backward(m)
    return _eddl.backward(m, target)


def optimize(l):
    return _eddl.optimize(l)


def update(m):
    """\
    Update the model weights.

    :param m: Model
    :return: None
    """
    return _eddl.update(m)


def print_loss(m, batch):
    """\
    Print model loss at the given batch.

    :param m: model
    :param batch: batch number
    :return: None
    """
    return _eddl.print_loss(m, batch)


# = Model constraints =

def clamp(m, min, max):
    """\
    Perform model parameter clamping between min and max.

    :param m: model
    :param min: minimum value
    :param max: maximum value
    :return: None
    """
    return _eddl.clamp(m, min, max)


# = Loss and metrics methods =

def compute_loss(L):
    """\
    Compute the loss of the associated model.

    :param L: loss object
    :return: computed loss
    """
    return _eddl.compute_loss(L)


def compute_metric(L):
    """\
    Compute the loss of the associated model (alias for ``compute_loss``).

    :param L: loss object
    :return: computed loss
    """
    return _eddl.compute_metric(L)


def getLoss(type_):
    """\
    Get loss by name.

    :param type_: loss name
    :return: loss
    """
    return _eddl.getLoss(type_)


def newloss(f, in_, name):
    """\
    Create a new loss.

    ``f`` can be a Layer -> Layer function (and ``in_`` must be a Layer) or a
    [Layer] -> Layer function (and ``in_`` must be a [Layer]).

    :param f: loss function
    :param in_: loss input
    :param name: loss name
    :return: loss
    """
    # core module has multiple overloads for this:
    #  1. newloss(f, in_, name)  where:
    #    f is a function [Layer] -> Layer
    #    in is a [Layer]
    #  2. newloss(f, in_, name)  where:
    #    f is a function Layer -> Layer
    #    in is a Layer
    return _eddl.newloss(f, in_, name)


def getMetric(type_):
    """\
    Get Metric by name.

    :param type_: Metric name
    :return: metric
    """
    return _eddl.getMetric(type_)


# def newmetric(f, in_, name)  # TODO


def detach(l):
    """\
    Set a layer as detached, excluding it from gradient computation.

    :param l: layer or list of layers to detach
    :return: detached layer(s)
    """
    # core module has multiple overloads for this:
    #  1. detach(l)  where l is a Layer and the return value is a Layer
    #  2. detach(l)  where l is a [Layer] and the return value is a [Layer]
    return _eddl.detach(l)


# == LAYERS ==

# = Core layers =

def Activation(parent, activation, params=[], name=""):
    """\
    Apply an activation function to the given layer.

    :param parent: parent layer
    :param activation: name of the activation function
    :param params: list of floats (parameters of the activation function)
    :param name: name of the output layer
    :return: Activation layer
    """
    return _eddl.Activation(parent, activation, params, name)


def Softmax(parent, name=""):
    """\
    Apply a softmax activation function to the given layer.

    :param parent: parent layer
    :param name: name of the output layer
    :return: softmax Activation layer
    """
    return _eddl.Softmax(parent, name)


def Sigmoid(parent, name=""):
    """\
    Apply a sigmoid activation function to the given layer.

    :param parent: parent layer
    :param name: name of the output layer
    :return: sigmoid Activation layer
    """
    return _eddl.Sigmoid(parent, name)


def HardSigmoid(parent, name=""):
    """\
    Apply a hard sigmoid activation function to the given layer.

    :param parent: parent layer
    :param name: name of the output layer
    :return: hard sigmoid Activation layer
    """
    return _eddl.HardSigmoid(parent, name)


def ReLu(parent, name=""):
    """\
    Apply a rectified linear unit activation function to the given layer.

    :param parent: parent layer
    :param name: name of the output layer
    :return: ReLu Activation layer
    """
    return _eddl.ReLu(parent, name)


def ThresholdedReLu(parent, alpha=1.0, name=""):
    """\
    Apply the thresholded version of a rectified linear unit activation
    function to the given layer.

    :param parent: parent layer
    :param name: name of the output layer
    :return: thresholded ReLu Activation layer
    """
    return _eddl.ThresholdedReLu(parent, alpha, name)


def LeakyReLu(parent, alpha=0.01, name=""):
    """\
    Apply the leaky version of a rectified linear unit activation
    function to the given layer.

    :param parent: parent layer
    :param name: name of the output layer
    :return: leaky ReLu Activation layer
    """
    return _eddl.LeakyReLu(parent, alpha, name)


def Elu(parent, alpha=1.0, name=""):
    """\
    Apply the exponential linear unit activation function to the given layer.

    :param parent: parent layer
    :param alpha: ELU coefficient
    :param name: name of the output layer
    :return: ELU Activation layer
    """
    return _eddl.Elu(parent, alpha, name)


def Selu(parent, name=""):
    """\
    Apply the scaled exponential linear unit activation function to the given
    layer.

    :param parent: parent layer
    :param name: name of the output layer
    :return: SELU Activation layer
    """
    return _eddl.Selu(parent, name)


def Exponential(parent, name=""):
    """\
    Apply the exponential (base e) activation function to the given layer.

    :param parent: parent layer
    :param name: name of the output layer
    :return: exponential Activation layer
    """
    return _eddl.Exponential(parent, name)


def Softplus(parent, name=""):
    """\
    Apply the softplus activation function to the given layer.

    :param parent: parent layer
    :param name: name of the output layer
    :return: softplus Activation layer
    """
    return _eddl.Softplus(parent, name)


def Softsign(parent, name=""):
    """\
    Apply the softsign activation function to the given layer.

    :param parent: parent layer
    :param name: name of the output layer
    :return: softsign Activation layer
    """
    return _eddl.Softsign(parent, name)


def Linear(parent, alpha=1.0, name=""):
    """\
    Apply the linear activation function to the given layer.

    :param parent: parent layer
    :param alpha: linear coefficient
    :param name: name of the output layer
    :return: linear Activation layer
    """
    return _eddl.Linear(parent, alpha, name)


def Tanh(parent, name=""):
    """\
    Apply the hyperbolic tangent activation function to the given layer.

    :param parent: parent layer
    :param name: name of the output layer
    :return: hyperbolic tangent Activation layer
    """
    return _eddl.Tanh(parent, name)


def Conv(parent, filters, kernel_size, strides=[1, 1], padding="same",
         use_bias=True, groups=1, dilation_rate=[1, 1], name=""):
    """\
    2D convolution layer.

    :param parent: parent layer
    :param filters: dimensionality of the output space (i.e., the number of
      output filters in the convolution)
    :param kernel_size: list of 2 integers, specifying the height and width of
      the 2D convolution window.
    :param strides: list of 2 integers, specifying the strides of the
      convolution along the height and width
    :param padding: one of "none", "valid" or "same"
    :param use_bias: whether the layer uses a bias vector
    :param groups: number of blocked connections from input to output channels
    :param dilation_rate: list of 2 integers, specifying the dilation rate
      to use for dilated convolution
    :param name: name of the output layer
    :return: Convolution layer
    """
    return _eddl.Conv(parent, filters, kernel_size, strides, padding,
                      use_bias, groups, dilation_rate, name)


def Conv1D(parent, filters, kernel_size, strides=[1], padding="same",
           use_bias=True, groups=1, dilation_rate=[1], name=""):
    """\
    1D convolution layer.

    :param parent: parent layer
    :param filters: dimensionality of the output space (i.e., the number of
      output filters in the convolution)
    :param kernel_size: list of 1 integer.
    :param strides: list of 1 integers
    :param padding: one of "none", "valid" or "same"
    :param use_bias: whether the layer uses a bias vector
    :param groups: number of blocked connections from input to output channels
    :param dilation_rate: list of 1 integer, specifying the dilation rate
      to use for dilated convolution
    :param name: name of the output layer
    :return: Convolution layer
    """
    return _eddl.Conv1D(parent, filters, kernel_size, strides, padding,
                        use_bias, groups, dilation_rate, name)


def Dense(parent, ndim, use_bias=True, name=""):
    """\
    Regular densely-connected layer.

    :param parent: parent layer
    :param ndim: dimensionality of the output space
    :param use_bias: whether the layer uses a bias vector
    :param name: name of the output layer
    :return: Dense layer
    """
    return _eddl.Dense(parent, ndim, use_bias, name)


def Dropout(parent, rate, iw=True, name=""):
    """\
    Apply dropout to a layer.

    The dropout consists of randomly setting a fraction of input units to
    0 at each update during training time, which helps prevent overfitting.

    :param parent: parent layer
    :param rate: fraction of input units to drop (between 0 and 1)
    :param iw: whether to perform weighting in inference
    :param name: name of the output layer
    :return: Dropout layer
    """
    return _eddl.Dropout(parent, rate, iw, name)


def Input(shape, name=""):
    """\
    Create a layer that can be used as input to a model.

    :param shape: list of dimensions, not including the batch size. For
      instance, shape=[32] indicates that the expected input will be batches
      of 32-dimensional vectors
    :param name: name of the output layer
    :return: Input layer
    """
    return _eddl.Input(shape, name)


def UpSampling(parent, size, interpolation="nearest", name=""):
    """\
    Upsampling layer.

    Similar to the scale transformation. The only difference is that
    upsampling repeats its rows/columns n times, while scaling uses a
    proportion.

    :param parent: parent layer
    :param size: list of 2 integers (upsampling factors for rows and columns)
    :param interpolation: "nearest" or "bilinear"
    :param name: name of the output layer
    :return: UpSampling layer
    """
    return _eddl.UpSampling(parent, size, interpolation, name)


def Reshape(parent, shape, name=""):
    """\
    Reshape an output to the given shape.

    :param parent: parent layer
    :param shape: target shape as a list of integers, not including the batch
      axis
    :param name: name of the output layer
    :return: Reshape layer
    """
    return _eddl.Reshape(parent, shape, name)


def Flatten(parent, name=""):
    """\
    Flatten the input. Does not affect the batch size.

    Equivalent to a :func:`.Reshape` with a shape of ``[-1]``.

    :param parent: parent layer
    :param name: name of the output layer
    :return: a Reshape layer
    """
    return _eddl.Flatten(parent, name)


def ConvT(parent, filters, kernel_size, output_padding, padding="same",
          dilation_rate=[1, 1], strides=[1, 1], use_bias=True, name=""):
    """\
    Transposed convolution layer (sometimes called deconvolution).

    The need for transposed convolutions generally arises from the desire to
    use a transformation going in the opposite direction of a normal
    convolution, i.e., from something that has the shape of the output of some
    convolution to something that has the shape of its input while maintaining
    a connectivity pattern that is compatible with said convolution.

    :param parent: parent layer
    :param filters: dimensionality of the output space (i.e., the number of
      output filters in the convolution)
    :param kernel_size: the height and width of the 2D convolution window
    :param output_padding: the amount of padding along the height and width of
      the output tensor. The amount of output padding along a given dimension
      must be lower than the stride along the same dimension
    :param padding: one of "valid" or "same"
    :param dilation_rate: the dilation rate to use for dilated convolution.
      Spacing between kernel elements
    :param strides: the strides of the convolution along the height and width
    :param use_bias: whether the layer uses a bias vector
    :param name: name of the output layer
    :return: ConvT layer

    """
    return _eddl.ConvT(parent, filters, kernel_size, output_padding, padding,
                       dilation_rate, strides, use_bias, name)


def Embedding(parent, vocsize, length, output_dim, mask_zeros=False, name=""):
    """\
    Turn positive integers (indexes) into dense vectors of fixed size. e.g.,
    ``[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]``.

    :param parent: parent layer
    :param vocsize: size of the vocabulary, i.e., maximum integer index + 1
    :param length: length of the sequence, to connect to Dense layers
      (non Recurrent)
    :param output_dim: dimension of the dense embedding
    :param name: name of the output layer
    :return: Embedding layer
    """
    return _eddl.Embedding(
        parent, vocsize, length, output_dim, mask_zeros, name
    )


def Transpose(parent, name=""):
    """\
    Transpose a Layer.

    :param parent: parent layer
    :param name: name of the output layer
    :return: the transposed layer
    """
    return _eddl.Transpose(parent, name)


# = Transformation layers =

def Crop(parent, from_coords, to_coords, reshape=True, constant=0.0, name=""):
    """\
    Crop the given image layer at ``[(top, left), (bottom, right)]``.

    :param parent: parent layer
    :param from_coords: [top, left] coordinates
    :param to_coords: [bottom, right] coordinates
    :param reshape: if True, the output shape will be new_shape (classical
      scale; recommended). If False, the output shape will be the input shape
      (scale < 100%: scale + padding; scale > 100%: crop + scale)
    :param constant: erasing value
    :param name: name of the output layer
    :return: Crop layer
    """
    return _eddl.Crop(parent, from_coords, to_coords, reshape, constant, name)


def CenteredCrop(parent, size, reshape=True, constant=0.0, name=""):
    """\
    Crop the given image layer at the center with size ``[width, height]``.

    :param parent: parent layer
    :param size: [height, width]
    :param reshape: If True, the output shape will be new_shape (classical
      scale; recommended). If False, the output shape will be the input shape
      (scale < 100%: scale + padding; scale > 100%: crop + scale)
    :param constant: erasing value
    :param name: name of the output layer
    :return: a Crop layer
    """
    return _eddl.CenteredCrop(parent, size, reshape, constant, name)


def CropScale(parent, from_coords, to_coords, da_mode="nearest", constant=0.0,
              name=""):
    """\
    Crop the given image layer at ``[(top, left), (bottom, right)]`` and scale
    it to the parent size.

    :param parent: parent layer
    :param from_coords: [top, left] coordinates
    :param to_coords: [bottom, right] coordinates
    :param da_mode: one of "nearest", "constant"
    :param constant: fill value for the area outside the rotated image, used
      for all channels
    :param name: name of the output layer
    :return: CropScale layer
    """
    return _eddl.CropScale(parent, from_coords, to_coords, da_mode, constant,
                           name)


def Cutout(parent, from_coords, to_coords, constant=0.0, name=""):
    """\
    Select a rectangle in an image layer at ``[(top, left), (bottom, right)]``
    and erase its pixels using a constant value.

    :param parent: parent layer
    :param from_coords: [top, left] coordinates
    :param to_coords: [bottom, right] coordinates
    :param constant: erasing value
    :param name: name of the output layer
    :return: Cutout layer
    """
    return _eddl.Cutout(parent, from_coords, to_coords, constant, name)


def Flip(parent, axis=0, name=""):
    """\
    Flip an image layer at the given axis.

    :param parent: parent layer
    :param axis: flip axis
    :param name: name of the output layer
    :return: Flip layer
    """
    return _eddl.Flip(parent, axis, name)


def HorizontalFlip(parent, name=""):
    """\
    Flip an image layer horizontally.

    :param parent: parent layer
    :param name: name of the output layer
    :return: a Flip layer
    """
    return _eddl.HorizontalFlip(parent, name)


def Rotate(parent, angle, offset_center=[0, 0], da_mode="original",
           constant=0.0, name=""):
    """\
    Rotate an image layer by the given angle, counterclockwise.

    :param parent: parent layer
    :param angle: rotation angle in degrees
    :param offset_center: center of rotation
    :param da_mode: one of "nearest", "constant"
    :param constant: fill value for the area outside the rotated image, used
      for all channels
    :return: Rotate layer
    """
    return _eddl.Rotate(parent, angle, offset_center, da_mode, constant, name)


def Scale(parent, new_shape, reshape=True, da_mode="nearest", constant=0.0,
          name=""):
    """\
    Resize an image layer to the given size as ``[height, width]``.

    :param parent: parent layer
    :param new_shape: new shape
    :param reshape: if True, the output shape will be new_shape (classical
      scale; recommended). If False, the output shape will be the input shape
      (scale < 100%: scale + padding; scale > 100%: crop + scale)
    :param da_mode: one of "nearest", "constant"
    :param constant: fill value for the area outside the resized image, used
      for all channels
    :return: Scale layer
    """
    return _eddl.Scale(parent, new_shape, reshape, da_mode, constant, name)


def Shift(parent, shift, da_mode="nearest", constant=0.0, name=""):
    """\
    Shift the input image.

    :param parent: parent layer
    :param shift: list of maximum absolute fraction for the horizontal and
      vertical translations
    :param da_mode: one of "nearest", "constant"
    :param constant: fill value for the area outside the resized image, used
      for all channels
    :return: Shift layer
    """
    return _eddl.Shift(parent, shift, da_mode, constant, name)


def VerticalFlip(parent, name=""):
    """\
    Flip an image layer vertically.

    :param parent: parent layer
    :param name: name of the output layer
    :return: a Flip layer
    """
    return _eddl.VerticalFlip(parent, name)


# = Data augmentation layers =

def RandomCrop(parent, new_shape, name=""):
    """\
    Crop an image layer at a random location with size ``[height, width]``.

    :param parent: parent layer
    :param new_shape: [height, width] size
    :param name: name of the output layer
    :return: CropRandom layer
    """
    return _eddl.RandomCrop(parent, new_shape, name)


def RandomCropScale(parent, factor, da_mode="nearest", name=""):
    """\
    Crop an image layer randomly and scale it to the parent size.

    :param parent: parent layer
    :param factor: crop range factor
    :param da_mode: one of "nearest", "constant"
    :param name: name of the output layer
    :return: CropScaleRandom layer
    """
    return _eddl.RandomCropScale(parent, factor, da_mode, name)


def RandomCutout(parent, factor_x, factor_y, constant=0.0, name=""):
    """\
    Randomly select a rectangle region in an image layer and erase its pixels.

    The random region is defined by the range ``[(min_x, max_x), (min_y,
    max_y)]`` (relative values).

    :param parent: parent layer
    :param factor_x: list of factors for horizontal size
    :param factor_y: list of factors for vertical size
    :param constant: erasing value
    :param name: name of the output layer
    :return: CutoutRandom layer
    """
    return _eddl.RandomCutout(parent, factor_x, factor_y, constant, name)


def RandomFlip(parent, axis, name=""):
    """\
    Flip an image layer at the given axis randomly.

    :param parent: parent layer
    :param axis: flip axis
    :param name: name of the output layer
    :return: FlipRandom layer
    """
    return _eddl.RandomFlip(parent, axis, name)


def RandomHorizontalFlip(parent, name=""):
    """\
    Flip an image layer horizontally, randomly.

    :param parent: parent layer
    :param name: name of the output layer
    :return: a FlipRandom layer
    """
    return _eddl.RandomHorizontalFlip(parent, name)


def RandomRotation(parent, factor, offset_center=[0, 0], da_mode="original",
                   constant=0.0, name=""):
    """\
    Rotate an image layer randomly.

    :param parent: parent layer
    :param factor: angle range in degrees (counterclockwise)
    :param offset_center: center of rotation
    :param da_mode: one of "original"
    :param constant: fill value for the area outside the rotated image, used
      for all channels.
    :param name: name of the output layer
    :return: RotateRandom layer
    """
    return _eddl.RandomRotation(parent, factor, offset_center, da_mode,
                                constant, name)


def RandomScale(parent, factor, da_mode="nearest", constant=0.0, name=""):
    """\
    Resize an image layer randomly.

    :param parent: parent layer
    :param factor: list of resize factors for the new shape
    :param da_mode: One of "nearest"
    :param constant: fill value for the area outside the resized image,
      used for all channels
    :param name: name of the output layer
    :return: ScaleRandom layer
    """
    return _eddl.RandomScale(parent, factor, da_mode, constant, name)


def RandomShift(parent, factor_x, factor_y, da_mode="nearest", constant=0.0,
                name=""):
    """\
    Shift an image layer randomly.

    :param parent: parent layer
    :param factor_x: list of factors for horizontal translations
    :param factor_y: list of factors for vertical translations
    :param da_mode: one of "nearest", "constant"
    :param constant: fill value for the area outside the resized image, used
      for all channels
    :param name: name of the output layer
    :return: ShiftRandom layer
    """
    return _eddl.RandomShift(parent, factor_x, factor_y, da_mode, constant,
                             name)


def RandomVerticalFlip(parent, name=""):
    """\
    Flip an image layer vertically, randomly.

    :param parent: parent layer
    :param name: name of the output layer
    :return: a RandomFlip layer
    """
    return _eddl.RandomVerticalFlip(parent, name)


# = Merge layers =

def Add(layers, name=""):
    """\
    Add input layers.

    :param layers: list of layers, all of the same shape
    :param name: name of the output layer
    :return: Add layer
    """
    return _eddl.Add(layers, name)


def Average(layers, name=""):
    """\
    Compute the average of a list of input layers.

    :param layers: list of layers, all of the same shape
    :param name: name of the output layer
    :return: Average layer
    """
    return _eddl.Average(layers, name)


def Concat(layers, axis=1, name=""):
    """\
    Concatenate input layers.

    :param layers: list of layers
    :param axis: axis along which to concatenate
    :param name: name of the output layer
    :return: Concat layer
    """
    return _eddl.Concat(layers, axis, name)


def MatMul(layers, name=""):
    return _eddl.MatMul(layers, name)


def Maximum(layers, name=""):
    """\
    Compute the maximum (element-wise) of a list of input layers.

    :param layers: list of layers, all of the same shape
    :param name: name of the output layer
    :return: Maximum layer
    """
    return _eddl.Maximum(layers, name)


def Minimum(layers, name=""):
    """\
    Compute the minimum (element-wise) of a list of input layers.

    :param layers: list of layers, all of the same shape
    :param name: name of the output layer
    :return: Minimum layer
    """
    return _eddl.Minimum(layers, name)


def Subtract(layers, name=""):
    """\
    Subtract two input layers.

    :param layers: list of two layers with the same shape
    :param name: name of the output layer
    :return: Substract layer
    """
    return _eddl.Subtract(layers, name)


# = Noise layers =

def GaussianNoise(parent, stddev, name=""):
    """\
    Apply additive zero-centered Gaussian noise.

    This is useful to mitigate overfitting (can be considered a form of
    random data augmentation). Gaussian Noise (GS) is a natural choice as
    corruption process for real valued inputs. Being a regularization layer,
    it is only active at training time.

    :param parent: parent layer
    :param stddev: standard deviation of the noise distribution
    :param name: name of the output layer
    :return: GaussianNoise layer
    """
    return _eddl.GaussianNoise(parent, stddev, name)


# = Normalization =

def BatchNormalization(parent, affine, momentum=0.9, epsilon=0.00001, name=""):
    """\
    Batch normalization layer.

    Normalize the activations of the input layer at each batch, i.e., apply a
    transformation that maintains the mean activation close to 0 and the
    activation standard deviation close to 1. See:
    https://arxiv.org/abs/1502.03167

    :param parent: parent layer
    :param affine: if True, this module has learnable affine parameters
    :param momentum: momentum for the moving mean and the moving variance
    :param epsilon: small float added to variance to avoid dividing by zero
    :param name: name of the output layer
    :return: BatchNorm layer
    """
    return _eddl.BatchNormalization(parent, affine, momentum, epsilon, name)


def LayerNormalization(parent, affine=True, epsilon=0.00001, name=""):
    """\
    Layer normalization layer.

    See: https://arxiv.org/abs/1607.06450.

    :param parent: parent layer
    :param affine: if True, this module has learnable affine parameters
    :param momentum: momentum for the moving mean and the moving variance
    :param epsilon: value added to the denominator for numerical stability
    :param name: name of the output layer
    :return: LayerNorm layer
    """
    return _eddl.LayerNormalization(parent, affine, epsilon, name)


def GroupNormalization(parent, groups, epsilon=0.001, affine=True, name=""):
    """\
    Group normalization layer.

    Divide the channels into groups and compute within each group the mean
    and variance for normalization. The computation is independent of batch
    sizes. See: https://arxiv.org/abs/1803.08494.

    :param parent: parent layer
    :param groups: number of groups in which the channels will be divided
    :param momentum: momentum for the moving mean and the moving variance
    :param epsilon: value added to the denominator for numerical stability
    :param affine: if True, this module has learnable affine parameters
    :param name: name of the output layer
    :return: GroupNorm layer
    """
    return _eddl.GroupNormalization(parent, groups, epsilon, affine, name)


def Norm(parent, epsilon=0.001, name=""):
    return _eddl.Norm(parent, epsilon, name)


def NormMax(parent, epsilon=0.001, name=""):
    return _eddl.NormMax(parent, epsilon, name)


def NormMinMax(parent, epsilon=0.001, name=""):
    return _eddl.NormMinMax(parent, epsilon, name)


# = Operator layers =

def Abs(l):
    """\
    Compute the element-wise absolute value of the given input layer.

    :param l: parent layer
    :return: Abs layer
    """
    return _eddl.Abs(l)


def Diff(l1, l2):
    """\
    Compute the difference between two layers or floats.

    :param l1: a layer or float
    :param l2: a layer or float
    :return: Diff layer
    """
    # l1, l2 can be either layers or floats
    return _eddl.Diff(l1, l2)


def Div(l1, l2):
    """\
    Compute the element-wise division of two layers or floats.

    :param l1: a layer or float
    :param l2: a layer or float
    :return: Div layer
    """
    # l1, l2 can be either layers or floats
    return _eddl.Div(l1, l2)


def Exp(l):
    """\
    Compute the element-wise exponential of the input layer.

    :param l: parent layer
    :return: Exp layer
    """
    return _eddl.Exp(l)


def Log(l):
    """\
    Compute the logarithm of the input layer.

    :param l: parent layer
    :return: Log layer
    """
    return _eddl.Log(l)


def Log2(l):
    """\
    Compute the base 2 logarithm of the input layer.

    :param l: parent layer
    :return: Log2 layer
    """
    return _eddl.Log2(l)


def Log10(l):
    """\
    Compute the base 10 logarithm of the input layer.

    :param l: parent layer
    :return: Log10 layer
    """
    return _eddl.Log10(l)


def Mult(l1, l2):
    """\
    Compute the element-wise multiplication of two layers or floats.

    :param l1: a layer or float
    :param l2: a layer or float
    :return: Mult layer
    """
    # l1, l2 can be either layers or floats
    return _eddl.Mult(l1, l2)


def Pow(l1, l2):
    """\
    Compute the element-wise power of two layers, or a layer and a float.

    :param l1: a layer
    :param l2: a layer or float
    :return: Pow layer
    """
    # l2 can be either a layer or a float
    return _eddl.Pow(l1, l2)


def Sqrt(l):
    """\
    Compute the square root of a layer.

    :param l: parent layer
    :return: Sqrt layer
    """
    return _eddl.Sqrt(l)


def Sum(l1, l2):
    """\
    Compute the sum of two layers or floats.

    :param l1: a layer or float
    :param l2: a layer or float
    :return: Sum layer
    """
    # l1, l2 can be either layers or floats
    return _eddl.Sum(l1, l2)


def Select(l, indices, name=""):
    """\
    Create a new layer which indexes the input layer using the entries in
    ``indices``.

    :param l: parent layer
    :param indices: list of indices to be selected
    :param name: name of the output layer
    :return: Select layer
    """
    return _eddl.Select(l, indices, name)


def Permute(l, dims, name=""):
    """\
    Permute the dimensions of the input according to a given pattern.

    :param l: parent layer
    :param dims: permutation pattern, does not include the samples dimension
    :param name: name of the output layer
    :return: Permute layer
    """
    return _eddl.Permute(l, dims, name)


# = Reduction layers =

def ReduceMean(l, axis, keepdims=False):
    return _eddl.ReduceMean(l, axis, keepdims)


def ReduceVar(l, axis, keepdims=False):
    return _eddl.ReduceVar(l, axis, keepdims)


def ReduceSum(l, axis, keepdims=False):
    return _eddl.ReduceSum(l, axis, keepdims)


def ReduceMax(l, axis, keepdims=False):
    return _eddl.ReduceMax(l, axis, keepdims)


def ReduceMin(l, axis, keepdims=False):
    return _eddl.ReduceMin(l, axis, keepdims)


def ReduceArgMax(l, axis, keepdims=False):
    return _eddl.ReduceArgMax(l, axis, keepdims)


# = Generator layers =

def GaussGenerator(mean, stdev, size):
    return _eddl.GaussGenerator(mean, stdev, size)


def UniformGenerator(low, high, size):
    return _eddl.UniformGenerator(low, high, size)


# = Pooling layers =

def AveragePool(parent, pool_size=[2, 2], strides=[2, 2], padding="none",
                name=""):
    """\
    Perform average pooling.

    :param parent: parent layer
    :param pool_size: size of the average pooling windows
    :param strides: factors by which to downscale
    :param padding: one of "none", "valid" or "same"
    :param name: name of the output layer
    :return: AveragePool layer
    """
    return _eddl.AveragePool(parent, pool_size, strides, padding, name)


def GlobalMaxPool(parent, name=""):
    """\
    Perform global max pooling.

    :param parent: parent layer
    :param name: name of the output layer
    :return: a MaxPool layer
    """
    return _eddl.GlobalMaxPool(parent, name)


def GlobalAveragePool(parent, name=""):
    """\
    Perform global average pooling.

    :param parent: parent layer
    :param name: name of the output layer
    :return: an AveragePool layer
    """
    return _eddl.GlobalAveragePool(parent, name)


def MaxPool(parent, pool_size=[2, 2], strides=[2, 2], padding="none", name=""):
    """\
    Perform Max pooling.

    :param parent: parent layer
    :param pool_size: size of the max pooling windows
    :param strides: factor by which to downscale
    :param padding: one of "none", "valid" or "same"
    :param name: name of the output layer
    :return: MaxPool layer
    """
    return _eddl.MaxPool(parent, pool_size, strides, padding, name)


def MaxPool1D(parent, pool_size=[2], strides=[2], padding="none", name=""):
    """\
    Perform 1D Max pooling.

    :param parent: parent layer
    :param pool_size: size of the max pooling windows
    :param strides: factor by which to downscale
    :param padding: one of "none", "valid" or "same"
    :param name: name of the output layer
    :return: MaxPool1D layer
    """
    return _eddl.MaxPool1D(parent, pool_size, strides, padding, name)


# = Recurrent layers =

def RNN(parent, units, activation="tanh", use_bias=True, bidirectional=False,
        name=""):
    """\
    Fully-connected RNN where the output is to be fed back to input.

    :param parent: parent layer
    :param units: dimensionality of the output space.
    :param activation: activation
    :param use_bias: whether the layer uses a bias vector
    :param bidirectional: whether the RNN is bidirectional
    :param name: name of the output layer
    :return: RNN layer
    """
    return _eddl.RNN(parent, units, activation, use_bias, bidirectional, name)


def LSTM(parent, units, mask_zeros=False, bidirectional=False, name=""):
    """\
    Long Short-Term Memory layer - Hochreiter 1997.

    :param parent: parent layer
    :param units: dimensionality of the output space.
    :param mask_zeros: boolean
    :param bidirectional: whether the net is bidirectional or not
    :param name: name of the output layer
    :return: LSTM layer
    """
    return _eddl.LSTM(parent, units, mask_zeros, bidirectional, name)


def Decoder(l, ld, op="concat"):
    return _eddl.Decoder(l, ld, op)


# = Layers methods =

def set_trainable(l, val):
    return _eddl.set_trainable(l, val)


def getOut(net):
    return _eddl.getOut(net)


# = Manage tensors inside layers =

def getOutput(l1):
    return _eddl.getOutput(l1)


def getDelta(l1):
    return _eddl.getDelta(l1)


def getParam(l1, p):
    return _eddl.getParam(l1, p)


def getGradient(l1, p):
    return _eddl.getGradient(l1, p)


def getParams(l1):
    return _eddl.getParams(l1)


def getGradients(l1):
    return _eddl.getGradients(l1)


def copyOutput(l1, l2):
    return _eddl.copyOutput(l1, l2)


def copyDelta(l1, l2):
    return _eddl.copyDelta(l1, l2)


def copyParam(l1, l2, p):
    return _eddl.copyParam(l1, l2, p)


def copyGradient(l1, l2, p):
    return _eddl.copyGradient(l1, l2, p)


# == INITIALIZERS ==

def GlorotNormal(l, seed=1234):
    """\
    Glorot normal initializer, also called Xavier normal initializer.

    It draws samples from a truncated normal distribution centered on 0 with
    ``stddev = sqrt(2 / (fan_in + fan_out))`` where fan_in is the number of
    input units in the weight tensor and fan_out is the number of output units
    in the weight tensor.

    :param l: parent layer to initialize
    :param seed: used to seed the random generator
    :return: GlorotNormal layer
    """
    return _eddl.GlorotNormal(l, seed)


def GlorotUniform(l, seed=1234):
    """\
    Glorot uniform initializer, also called Xavier uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit] where
    limit is ``sqrt(6 / (fan_in + fan_out))``, where fan_in is the number of
    input units in the weight tensor and fan_out is the number of output units
    in the weight tensor.

    :param l: parent layer to initialize
    :param seed: used to seed the random generator
    :return: GlorotUniform layer
    """
    return _eddl.GlorotUniform(l, seed)


def HeNormal(l, seed=1234):
    """\
    He normal initializer.

    It draws samples from a truncated normal distribution centered on 0 with
    stddev = sqrt(2 / (fan_in)) where fan_in is the number of input units in
    the weight tensor.

    :param l: parent layer to initialize
    :param seed: used to seed the random generator
    :return: HeNormal layer
    """
    return _eddl.HeNormal(l, seed)


def HeUniform(l, seed=1234):
    """\
    He uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit] where
    limit is sqrt(6 / (fan_in )) where fan_in is the number of input units in
    the weight tensor.

    :param l: parent layer to initialize
    :param seed: used to seed the random generator
    :return: HeUniform layer
    """
    return _eddl.HeUniform(l, seed)


def RandomNormal(l, m=0.0, s=0.1, seed=1234):
    """\
    Random normal initializer.

    :param l: parent layer to initialize
    :param m: mean of the normal distribution to draw samples
    :param s: standard deviation of the normal distribution to draw samples
    :param seed: used to seed the random generator
    :return: RandomNormal layer
    """
    return _eddl.RandomNormal(l, m, s, seed)


def RandomUniform(l, min=0.0, max=0.1, seed=1234):
    """\
    Random uniform initializer.

    :param l: parent layer to initialize
    :param min: min of the distribution
    :param max: max of the distribution
    :param seed: used to seed the random generator
    :return: RandomUniform layer
    """
    return _eddl.RandomUniform(l, min, max, seed)


def Constant(l, v=0.1):
    """\
    Initializer that generates tensors initialized to a constant value.

    :param l: parent layer to initialize
    :param v: value of the generator
    :return: Constant layer
    """
    return _eddl.Constant(l, v)


# == REGULARIZERS ==

def L2(l, l2):
    """\
    Regularizer for L2 regularization.

    :param l: parent layer to regularize
    :param l2: L2 regularization factor
    :return: the input layer, regularized
    """
    return _eddl.L2(l, l2)


def L1(l, l1):
    """\
    Regularizer for L1 regularization.

    :param l: parent layer to regularize
    :param l1: L1 regularization factor
    :return: the input layer, regularized
    """
    return _eddl.L1(l, l1)


def L1L2(l, l1, l2):
    """\
    Regularizer for L1 and L2 regularization.

    :param l: parent layer to regularize
    :param l1: L1 regularization factor
    :param l2: L2 regularization factor
    :return: the input layer, regularized
    """
    return _eddl.L1L2(l, l1, l2)


# == DATASETS ==

def exist(name):
    return _eddl.exist(name)


def download_mnist():
    """\
    Download the MNIST dataset.

    See: http://yann.lecun.com/exdb/mnist/
    """
    return _eddl.download_mnist()


def download_cifar10():
    """\
    Download the CIFAR-10 Dataset.

    See: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    return _eddl.download_cifar10()


def download_drive():
    """\
    Download the DRIVE Dataset.

    See: https://drive.grand-challenge.org/
    """
    return _eddl.download_drive()


def download_imdb_2000():
    """\
    Download the IMDB Dataset, 2000 most frequent words.

    See: https://ai.stanford.edu/~amaas/data/sentiment/
    """
    return _eddl.download_imdb_2000()


def download_eutrans():
    """\
    Download the EuTrans Dataset.
    """
    return _eddl.download_eutrans()


def download_flickr():
    """\
    Download the Flickr Dataset (small partition).
    """
    return _eddl.download_flickr()


# == ONNX ==

def save_net_to_onnx_file(net, path):
    return _eddl.save_net_to_onnx_file(net, path)


def import_net_from_onnx_file(path):
    return _eddl.import_net_from_onnx_file(path)


def serialize_net_to_onnx_string(net, gradients):
    return _eddl.serialize_net_to_onnx_string(net, gradients)


def import_net_from_onnx_string(model_string):
    return _eddl.import_net_from_onnx_string(model_string)
