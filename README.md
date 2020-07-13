<div align="center">
  <img src="https://raw.githubusercontent.com/deephealthproject/pyeddl/master/docs/logo.png" height="220" width="185">
</div>

-----------------


**PyEDDL** is a Python wrapper for [EDDL](https://github.com/deephealthproject/eddl), the European Distributed Deep Learning library.

The documentation is available at https://deephealthproject.github.io/pyeddl.

As a preview, here is a simple neural network training example:

```python
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

def main():
    eddl.download_mnist()

    epochs = 10
    batch_size = 100
    num_classes = 10

    in_ = eddl.Input([784])
    layer = in_
    layer = eddl.LeakyReLu(eddl.Dense(layer, 1024))
    layer = eddl.LeakyReLu(eddl.Dense(layer, 1024))
    layer = eddl.LeakyReLu(eddl.Dense(layer, 1024))
    out = eddl.Softmax(eddl.Dense(layer, num_classes))
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.rmsprop(0.01),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU()
    )

    x_train = Tensor.load("mnist_trX.bin")
    y_train = Tensor.load("mnist_trY.bin")
    x_test = Tensor.load("mnist_tsX.bin")
    y_test = Tensor.load("mnist_tsY.bin")
    x_train.div_(255.0)
    x_test.div_(255.0)

    eddl.fit(net, [x_train], [y_train], batch_size, epochs)
    eddl.evaluate(net, [x_test], [y_test])

if __name__ == "__main__":
    main()
```

If you're interested in contributing to the development, see the
[contributing](CONTRIBUTING.md) docs. They contain information on how to
generate the automated part of the bindings, build binary wheels, etc.
