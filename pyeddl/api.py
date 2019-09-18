import io
from . import _core, utils

DEV_CPU = 0
DEV_GPU = 1000
DEV_FPGA = 2000

__all__ = [
    "Input",
    "Activation",
    "Dense",
    "Model",
    "sgd",
    "CS_CPU",
    "build",
    "T_load",
    "div",
    "fit",
    "evaluate",
    "GaussianNoise",
    "Reshape",
    "MaxPool",
    "Conv",
    "save",
    "load",
]


def Input(shape, name=""):
    t = _core.Tensor([1] + shape, DEV_CPU)
    return _core.LInput(t, name, DEV_CPU)


def Activation(parent, activation, name=""):
    return _core.LActivation(parent, activation, name, DEV_CPU)


def Dense(parent, ndim, use_bias=True, name=""):
    return _core.LDense(parent, ndim, use_bias, name, DEV_CPU)


def Model(in_, out):
    return _core.Net(in_, out)


# optimizer
def sgd(lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
    return _core.SGD(lr, momentum, weight_decay, nesterov)


# compserv
def CS_CPU(threads, lsb=1):
    return _core.CompServ(threads, [], [], lsb)


def CS_GPU(g, lsb=1):
    return _core.CompServ(0, g, [], lsb)


def CS_FPGA(f, lsb=1):
    return _core.CompServ(0, [], f, lsb)


def build(model, optimizer, losses, metrics, compserv):
    losses = [utils.loss_func(_) for _ in losses]
    metrics = [utils.metric_func(_) for _ in metrics]
    model.build(optimizer, losses, metrics, compserv)


def T_load(fname):
    return _core.LTensor(fname)


def div(ltensor, v):
    ltensor.input.div(v)


def fit(model, inputs, outputs, batch_size, n_epochs):
    inputs = [_.input for _ in inputs]
    outputs = [_.input for _ in outputs]
    model.fit(inputs, outputs, batch_size, n_epochs)


def evaluate(model, inputs, outputs):
    inputs = [_.input for _ in inputs]
    outputs = [_.input for _ in outputs]
    model.evaluate(inputs, outputs)


def GaussianNoise(parent, stddev, name=""):
    return _core.LGaussianNoise(parent, stddev, name, DEV_CPU)


def Reshape(parent, shape, name=""):
    return _core.LReshape(parent, [1] + shape, name, DEV_CPU)


def MaxPool(parent, pool_size=None, strides=None, padding="none", name=""):
    if pool_size is None:
        pool_size = [2, 2]
    if strides is None:
        strides = [2, 2]
    pd = _core.PoolDescriptor(pool_size, strides, padding)
    return _core.LMaxPool(parent, pd, name, DEV_CPU)


def Conv(parent, filters, kernel_size, strides=None, padding="same",
         groups=1, dilation_rate=None, use_bias=True, name=""):
    if strides is None:
        strides = [1, 1]
    if dilation_rate is None:
        dilation_rate = [1, 1]
    return _core.LConv(parent, filters, kernel_size, strides, padding, groups,
                       dilation_rate, use_bias, name, DEV_CPU)


def save(model, fname):
    with io.open(fname, "wb") as f:
        model.save(f)


def load(model, fname):
    with io.open(fname, "rb") as f:
        model.load(f)
