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
def CS_CPU(threads):
    return _core.CompServ(threads, [], [])


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
