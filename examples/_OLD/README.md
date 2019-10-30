# Examples

This directory contains a series of usage examples. Most of them are Python
portings of C++ examples from EDDL.


## Custom metrics and GPU runs

The `eddl_ae_py_metric` example shows how to use a custom metric written in
Python, `py_mse`, which is simply a Python implementation of the MSE metric
available in EDDL:

```python
def py_mse(t, y):
    aux = Tensor(t.getShape(), t.device)
    Tensor.add(1, t, -1, y, aux, 0)
    Tensor.el_mult(aux, aux, aux, 0)
    return aux.sum()
```

The implementation uses the Python bindings for Tensor methods, and works both
in the CPU and in the GPU case. While implementing `py_mse` with NumPy is
possible, when running on GPU tensors must be copied:

```python
def py_mse(t, y):
    tc = Tensor(t.getShape(), DEV_CPU)
    Tensor.copy(t, tc)
    yc = Tensor(y.getShape(), DEV_CPU)
    Tensor.copy(y, yc)
    a = np.array(tc, copy=False)
    b = np.array(yc, copy=False)
    return np.sum(np.square(a - b))
```

Simply doing `a = np.array(t)` would not work, since tensor data is stored on
the GPU memory.
