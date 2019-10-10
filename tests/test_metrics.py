from pyeddl._core import Tensor, CustomMetric, MMeanSquaredError


class PyMSE(object):

    def value(t, y):
        aux = Tensor(t.getShape(), t.device)
        Tensor.add(1, t, -1, y, aux, 0)
        Tensor.el_mult(aux, aux, aux, 0)
        return aux.sum()


def test_custom_metric():
    T = Tensor.ones([3, 4], 0)
    Y = Tensor([3, 4], 0)
    Y.set(0.15)
    m = CustomMetric(PyMSE, "py_mean_squared_error")
    v = m.value(T, Y)
    assert v == MMeanSquaredError().value(T, Y)
