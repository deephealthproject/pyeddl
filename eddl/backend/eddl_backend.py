import numpy as np

import eddl
from eddl import _C
from eddl import losses
from eddl import metrics


def _get_optim(optim):
    if isinstance(optim, eddl.optim.SGD):
        return _C.SGD(optim.lr, optim.momentum, optim.decay, optim.nesterov)
    else:
        NotImplementedError('optim')


def _get_loss(loss):
    if isinstance(loss, losses.MeanSquaredError):
        return _C.LMeanSquaredError()
    elif isinstance(loss, losses.CategoricalCrossEntropy):
        return _C.LCrossEntropy()
    elif isinstance(loss, losses.CategoricalSoftCrossEntropy):
        return _C.LSoftCrossEntropy()
    else:
        NotImplementedError('Unknown loss')


def _get_metric(metric):
    if isinstance(metric, metrics.MeanSquaredError):
        return _C.MMeanSquaredError()
    elif isinstance(metric, metrics.CategoricalAccuracy):
        return _C.MCategoricalAccuracy()
    else:
        NotImplementedError('Unknown metric')


def _get_compserv(device, workers=1):
    if device == 'cpu':
        return _C.EDDL.CS_CPU(workers)
    elif device == 'gpu':
        NotImplementedError('GPU')
    elif device == 'fpga':
        NotImplementedError('FPGA')
    else:
        NotImplementedError('Unknown device')


def get_model(name='mlp', batch_size=128):
    if name == 'mlp':
        return _C.EDDL.get_model_mlp(batch_size)
    elif name == 'cnn':
        return _C.EDDL.get_model_cnn(batch_size)
    else:
        NotImplementedError('Unknown model')


def compile(model, optim, losses, metrics, device, workers):
    model.c_optim = _get_optim(optim)
    model.c_losses = [_get_loss(l) for l in losses]
    model.c_metrics = [_get_metric(m) for m in metrics]
    model.c_compserv = _get_compserv(device, workers)

    _C.EDDL.build(model.c_model, model.c_optim, model.c_losses, model.c_metrics, model.c_compserv)


def summary(model):
    return model.c_model.summary()


def plot(model, filename):
    return model.c_model.plot(filename)


def train_batch(model, x, y):
    # Transform array to a single vector
    tx = _C.tensor_from_npy(x, _C.DEV_CPU)
    ty = _C.tensor_from_npy(y, _C.DEV_CPU)

    model.c_model.train_batch_ni([tx], [ty])

    # TODO: Fix loss and metric values
    # for i in range(len(model.tout)):
    #     total_loss += model.c_model.fiterr[i][0]
    #     total_metric += model.c_model.fiterr[i][1]
    return model.c_model.fiterr


def evaluate(model, x, y):
    # Transform array to a single vector
    tx = _C.tensor_from_npy(x, _C.DEV_CPU)
    ty = _C.tensor_from_npy(y, _C.DEV_CPU)

    return model.c_model.evaluate([tx], [ty])
