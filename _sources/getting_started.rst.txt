.. _getting_started:

Getting Started
===============

This section contains some PyEDDL coding examples. Refer to the :ref:`API docs
<api>` for additional details. You can find more examples in the ``examples``
directory of the `GitHub repo <https://github.com/deephealthproject/pyeddl>`_.


Creating and manipulating tensors
---------------------------------

Create an uninitialized tensor with a given shape:

.. code-block:: python

    import pyeddl.eddlT as eddlT
    shape = [3, 4]
    t = eddlT.create(shape)

Create a tensor with evenly spaced values and compute the element-wise cosine:

.. code-block:: python

    import math
    import pyeddl.eddlT as eddlT
    t = eddlT.linspace(0, math.pi, 8)
    cos_t = eddlT.cos(t)


To/from NumPy conversions
-------------------------

Create a tensor with data initialized from a NumPy array:

.. code-block:: python

    import pyeddl.eddlT as eddlT
    import numpy as np
    a = np.arange(12).reshape(3, 4).astype(np.float32)
    t = eddlT.create(a)

Convert a tensor to a NumPy array:

.. code-block:: python

    import pyeddl.eddlT as eddlT
    import numpy as np
    t = eddlT.ones([3, 4])  # [3, 4] tensor filled with ones
    a = np.array(t)

You can also do ``a = np.array(t, copy=False)`` to view the tensor data as an
array without copying the data. Another way to get the tensor data as an array
(with copy) is ``a = eddlT.getdata(t)``.

Load NumPy data into EDDL tensors:

.. code-block:: python

    from urllib.request import urlretrieve
    import numpy as np
    import pyeddl.eddlT as eddlT
    urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", "mnist.npz")
    with np.load("mnist.npz") as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    t_x_train = eddlT.create(x_train.astype(np.float32))
    t_y_train = eddlT.create(y_train.astype(np.float32))
    t_x_test = eddlT.create(x_test.astype(np.float32))
    t_y_test = eddlT.create(y_test.astype(np.float32))


Training a MLP network
----------------------

.. code-block:: python

    import pyeddl.eddl as eddl
    import pyeddl.eddlT as eddlT

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

        x_train = eddlT.load("trX.bin")
        y_train = eddlT.load("trY.bin")
        x_test = eddlT.load("tsX.bin")
        y_test = eddlT.load("tsY.bin")
        eddlT.div_(x_train, 255.0)
        eddlT.div_(x_test, 255.0)

        eddl.fit(net, [x_train], [y_train], batch_size, epochs)
        eddl.evaluate(net, [x_test], [y_test])

    if __name__ == "__main__":
        main()
