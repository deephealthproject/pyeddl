.. _getting_started:

Getting Started
===============

This section contains some PyEDDL coding examples. Refer to the :ref:`API docs
<api>` for additional details.


Creating and manipulating tensors
---------------------------------

Create an uninitialized tensor with a given shape:

.. code-block:: python

    from pyeddl.tensor import Tensor
    shape = [3, 4]
    t = Tensor(shape)

By default, the tensor stores its data on the CPU. If you are using a
GPU-enabled version of PyEDDL/EDDL, you can create a GPU tensor (i.e., with
data stored on the GPU):

.. code-block:: python

    from pyeddl.tensor import Tensor, DEV_GPU
    shape = [3, 4]
    t = Tensor(shape, DEV_GPU)

Create a tensor with evenly spaced values and compute the element-wise cosine:

.. code-block:: python

    import math
    from pyeddl.tensor import Tensor
    t = Tensor.linspace(0, math.pi, 8)
    cos_t = Tensor.cos(t)


To/from NumPy conversions
-------------------------

Create a tensor with data initialized from a NumPy array:

.. code-block:: python

    from pyeddl.tensor import Tensor
    import numpy as np
    a = np.arange(12).reshape(3, 4).astype(np.float32)
    t = Tensor.fromarray(a)
    t.print()
    t.info()

Get tensor data as a NumPy array:

.. code-block:: python

    from pyeddl.tensor import Tensor
    import numpy as np
    t = Tensor.ones([3, 4])  # [3, 4] tensor filled with ones
    a = t.getdata()

Note that ``getdata`` performs a copy of the tensor's data. If ``t`` is a CPU
tensor, you can also do ``a = np.array(t, copy=False)`` to view the tensor's
data as an array without any copying being done. In this case, any
modification of ``a`` is reflected in ``t``:

.. code-block:: python

    from pyeddl.tensor import Tensor
    import numpy as np
    t = Tensor.ones([3, 4])
    a = np.array(t, copy=False)
    t.print()
    a[1, 1:3] = 2
    t.print()

However, this is only possible for CPU tensors (NumPy does not support other
devices).

Load NumPy data into EDDL tensors:

.. code-block:: python

    from urllib.request import urlretrieve
    import numpy as np
    from pyeddl.tensor import Tensor
    urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", "mnist.npz")
    with np.load("mnist.npz") as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    t_x_train = Tensor.fromarray(x_train.astype(np.float32))
    t_y_train = Tensor.fromarray(y_train.astype(np.float32))
    t_x_test = Tensor.fromarray(x_test.astype(np.float32))
    t_y_test = Tensor.fromarray(y_test.astype(np.float32))


Training a MLP network
----------------------

.. code-block:: python

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


Additional examples
-------------------

The MLP training above is just one example of neural network training with
PyEDDL. Many more examples are available in the `examples directory of the
GitHub repository
<https://github.com/deephealthproject/pyeddl/tree/master/examples>`_. These
examples are Python portings of the `C++ EDDL examples
<https://github.com/deephealthproject/eddl/tree/master/examples>`_.
