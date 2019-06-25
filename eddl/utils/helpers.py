from eddl import losses
from eddl import metrics


def get_loss(loss):
    if isinstance(loss, str):
        if loss in {'mean_squared_error', 'mse'}:
            return losses.MeanSquaredError()
        elif loss in {'categorical_crossentropy', 'crossentropy'}:
            return losses.CategoricalCrossEntropy()
        elif loss in {'categorical_soft_crossentropy', 'soft_crossentropy'}:
            return losses.CategoricalSoftCrossEntropy()
        else:
            raise NotImplementedError('Unknown loss')
    elif isinstance(loss, losses.Loss):
        return loss
    else:
        raise TypeError('Unknown loss type')


def get_metric(metric):
    if isinstance(metric, str):
        if metric in {'mean_squared_error', 'mse'}:
            return metrics.MeanSquaredError()
        elif metric in {'categorical_accuracy', 'accuracy'}:
            return metrics.CategoricalAccuracy()
        else:
            NotImplementedError('Unknown metric')
    elif isinstance(metric, metrics.Metric):
        return metric
    else:
        raise TypeError('Unknown metric type')