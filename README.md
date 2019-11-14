<div align="center">
  <img src="https://raw.githubusercontent.com/deephealthproject/pyeddl/master/docs/logo-pyeddl.png" height="120" width="300">
</div>

-----------------


**PyEDDL** is a Python wrapper for [EDDL](https://github.com/deephealthproject/eddl), the European Distributed Deep Learning library.


## Quick start

    git clone --recurse-submodules https://github.com/deephealthproject/pyeddl.git
    cd pyeddl
    python3 -m pip install numpy pybind11 pytest
    export EDDL_WITH_CUDA="true"
    python3 setup.py install
    pytest tests
    

See [full installation instructions below](#installation).


## Getting started

```python
import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT


def main():
    eddl.download_mnist()

    epochs = 10
    batch_size = 100
    num_classes = 10

    in_ = eddl.Input([784])
    layer = in_
    layer = eddl.BatchNormalization(
        eddl.Activation(eddl.L2(eddl.Dense(layer, 1024), 0.0001), "relu")
    )
    layer = eddl.BatchNormalization(
        eddl.Activation(eddl.L2(eddl.Dense(layer, 1024), 0.0001), "relu")
    )
    layer = eddl.BatchNormalization(
        eddl.Activation(eddl.L2(eddl.Dense(layer, 1024), 0.0001), "relu")
    )
    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU(4)
    )
    print(eddl.summary(net))

    x_train = eddlT.load("trX.bin")
    y_train = eddlT.load("trY.bin")
    x_test = eddlT.load("tsX.bin")
    y_test = eddlT.load("tsY.bin")
    eddlT.div_(x_train, 255.0)
    eddlT.div_(x_test, 255.0)

    for i in range(epochs):
        eddl.fit(net, [x_train], [y_train], batch_size, 1)
        eddl.evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    main()
```

You can find more examples under `examples`.


## Installation

### Requirements

- Python 3
- EDDL
- NumPy
- pybind11
- pytest (if you want to run the tests)


### EDDL Installation

Complete EDDL installation instructions are available at
https://github.com/deephealthproject/eddl. Make sure you build EDDL with
shared library support. Here is a sample build sequence:

```
cd third_party/eddl
mkdir build
cd build
cmake -D EDDL_SHARED=ON ..
make
make install
```

Make sure EDDL installation artifacts are in "standard" system locations. You
might need to copy them from the `third_party/eddl/build/install` directory
created as a result of the EDDL installation process described above. For
instance:

```
cd third_party/eddl/build
cp -rf install/include/eddl /usr/include/
cp -rf install/include/third_party/eigen/Eigen /usr/include/
cp install/lib/libeddl.so /usr/lib/
```

### Enabling GPU acceleration

If EDDL was compiled for GPU, you need to export the `EDDL_WITH_CUDA`
environment variable **before installing PyEDDL** so that `setup.py` will also link the
`cudart`, `cublas` and `curand` libraries. Again, these will be expected in
"standard" system locations, so you might need to create symlinks depending on
your CUDA toolkit installation. For instance:

```
export EDDL_WITH_CUDA="true"
ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so /usr/lib/
ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcurand.so /usr/lib/
ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcublas.so /usr/lib/
```

### PyEDDL installation

Install PyEDDL as follows:

```
python3 -m pip install numpy pybind11 pytest
python3 setup.py install
```

Then, you can test your installation by running the PyEDDL tests:

    pytest tests
