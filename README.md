<div align="center">
  <img src="https://raw.githubusercontent.com/deephealthproject/pyeddl/master/docs/logo.png" height="220" width="185">
</div>

-----------------


**PyEDDL** is a Python wrapper for [EDDL](https://github.com/deephealthproject/eddl), the European Distributed Deep Learning library.

Each PyEDDL version requires a specific EDDL version:

PyEDDL version | EDDL version |
-------------- | ------------ |
0.1.0          | 0.2.2        |
0.2.0          | 0.3          |
0.3.0          | 0.3.1        |
0.4.0          | 0.4.2        |
0.5.0          | 0.4.3        |
0.6.0          | 0.4.4        |


## Quick start

The following assumes you have EDDL already installed in "standard"
system paths (e.g., `/usr/local/include`, `/usr/local/lib`). Note that you
need the shared library (`libeddl.so`).

    python3 -m pip install numpy pybind11 pytest
    python3 -m pip install pyeddl

See [full installation instructions below](#installation).


## Getting started

```python
import pyeddl.eddl as eddl
import pyeddl.eddlT as eddlT

def main():
    eddl.download_mnist()

    epochs = 10
    batch_size = 100
    num_classes = 10

    in_ = eddl.Input([784])
    layer = in_
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    layer = eddl.ReLu(eddl.Dense(layer, 1024))
    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU()
    )
    eddl.summary(net)

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
```

You can find more examples under `examples`.


## Loading NumPy data into EDDL tensors

To load NumPy data into tensors, load them as arrays with `numpy.load` and
then use `eddlT.create` to convert the arrays into tensors:

```python
from urllib.request import urlretrieve
import numpy as np
import pyeddl.eddlT as eddlT

urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", "mnist.npz")
with np.load("mnist.npz") as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
t_x_train = eddlT.create(x_train.astype(np.float32))
...
```

See `examples/Tensor/array_tensor_save.py` for a full example.


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
shared library support. Here is a sample build sequence from the pyeddl git
repository:

```
git clone --recurse-submodules https://github.com/deephealthproject/pyeddl.git
cd pyeddl
cd third_party/eddl
mkdir build
cd build
cmake -D BUILD_SHARED_LIB=ON -D BUILD_PROTOBUF=ON ..
make
make install
```

**NOTE:** recent (>=0.4) versions of EDDL do not include Eigen as a git
submodule anymore, so it must be installed as a dependency. For instance,
`apt-get install libeigen3-dev`. Also note that EDDL code includes Eigen
headers like in this example: `#include <Eigen/Dense>`, e.g., with `Eigen` as
the root directory. However, Eigen installations usually have the header
rooted at `eigen3` (for instance, the apt installation places them in
`/usr/include/eigen3`). To work around this you can either add a symlink or
set `CPATH`, e.g., `export CPATH="/usr/include/eigen3:${CPATH}"`. Finally, the
current version of Eigen installed by apt has [issues with
CUDA](https://devtalk.nvidia.com/default/topic/1026622/nvcc-can-t-compile-code-that-uses-eigen). If
you are compiling for GPU, install a recent version of Eigen from source. See
`Dockerfile.eddl` and `Dockerfile.eddl-gpu` for more details.

**NOTE:** EDDL versions 0.3 and 0.3.1 are affected by an issue that breaks the
building of the shared library. To work around this, you can patch the EDDL
code with `eddl_0.3.patch` (at the top level in the pyeddl git repository):

```
cd third_party/eddl
git apply ../../eddl_0.3.patch
mkdir build
cd build
cmake -D BUILD_SHARED_LIB=ON -D BUILD_PROTOBUF=ON ..
make
make install
```


### Enabling GPU acceleration

If EDDL was compiled for GPU, you need to export the `EDDL_WITH_CUDA`
environment variable **before installing PyEDDL** so that `setup.py` will also
link the `cudart`, `cublas` and `curand` libraries. These will be
expected in "standard" system locations, so you might need to create symlinks
depending on your CUDA toolkit installation. For instance:

```
export EDDL_WITH_CUDA="true"
ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so /usr/lib/
ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcurand.so /usr/lib/
ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcublas.so /usr/lib/
```

### PyEDDL installation

Install requirements:

```
python3 -m pip install numpy pybind11 pytest
```

Install PyEDDL:

```
python3 -m pip install pyeddl
```

Or, from the pyeddl git repository:

```
python3 setup.py install
```

Then, you can test your installation by running the PyEDDL tests. From the
pyeddl git repository:

    pytest tests


### Disabling unwanted modules

By default, PyEDDL assumes a complete EDDL installation, including optional
modules, and builds bindings for all of them. You can disable support for
specific modules via environment variables. For instance, suppose you
installed EDDL without protobuf support: by default, PyEDDL will try to build
the bindings for protobuf-specific EDDL tools. To avoid this, set the
`EDDL_WITH_PROTOBUF` environment variable to `OFF` (or `FALSE`) before
building PyEDDL.


### EDDL installed in an arbitrary directory

The above installation instructions assume EDDL has been installed in standard
system paths. However, EDDL supports installation in an arbitrary directory,
for instance:

```
cd third_party/eddl
mkdir build
cd build
cmake -D BUILD_SHARED_LIB=ON -D BUILD_PROTOBUF=ON -DCMAKE_INSTALL_PREFIX=/home/myuser/eddl ..
make
make install
```

You can tell the PyEDDL setup script about this via the EDDL_DIR environment
variable:

```
export EDDL_DIR=/home/myuser/eddl
python3 setup.py install
```

In this way, `setup.py` will look for additional include files in
`/home/myuser/eddl/include` and for additional libraries in
`/home/myuser/eddl/lib`.
