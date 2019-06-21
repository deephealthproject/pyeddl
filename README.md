<div align="center">
  <img src="https://raw.githubusercontent.com/salvacarrion/salvacarrion.github.io/master/assets/hot-linking/logo-pyeddl.png" height="120" width="300">
</div>

-----------------

[![Documentation Status](https://readthedocs.org/projects/pyeddl/badge/?version=latest)](https://pyeddl.readthedocs.io/en/latest/?badge=latest) 
[![Build Status](https://travis-ci.org/salvacarrion/pyeddl.svg?branch=master)](https://travis-ci.org/salvacarrion/pyeddl)
[![codecov](https://codecov.io/gh/salvacarrion/pyeddl/branch/master/graph/badge.svg)](https://codecov.io/gh/salvacarrion/pyeddl)
[![Gitter chat](https://badges.gitter.im/USER/pyeddl.png)](https://gitter.im/pyeddl "Gitter chat")

**PyEDDL** is a Python package that wraps the EDDL library in order to provide two high-level features:
- Tensor computation (like NumPy) with strong GPU acceleration
- Deep neural networks

> **What is EDDL?** A European Distributed Deep Learning library for numerical computation tailored to the healthcare domain.
> Repo: [https://github.com/deephealthproject/eddl](https://github.com/deephealthproject/eddl)

# Requirements

- Python 3
- CMake 3.9.2 or higher
- A modern compiler with C++11 support

To clone all third_party submodules use:

```bash
git clone --recurse-submodules -j8 https://github.com/deephealthproject/pyeddl.git
```
 

# Installation

To build and install `pyeddl`, clone or download this repository and then, from within the repository, run:

```bash
python3 setup.py install
```


# Getting started

```python
import pyeddl
from pyeddl.model import Model
from pyeddl.datasets import mnist
from pyeddl.utils import to_categorical

# Params
batch_size = 1000
num_classes = 10
epochs = 1

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# Reshape training dataset
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))

# Transform to categorical
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# View model
m = Model.from_model('mlp', batch_size=batch_size)
print(m.summary())
m.plot("model.pdf")

# Building params
optim = pyeddl.optim.SGD(0.01, 0.9)
losses = ['soft_crossentropy']
metrics = ['accuracy']

# Build model
m.compile(optimizer=optim, losses=losses, metrics=metrics, device='cpu', workers=4)

# Training
m.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Evaluate
print("Evaluate train:")
m.evaluate(x_test, y_test)
```

Learn more examples about how to do specific tasks in PyEDDL at the [tutorials page](https://pyeddl.readthedocs.io/en/latest/user/tutorial.html)


# Tests

To execute all unit tests, run the following command:

```bash
python3 setup.py test
```
