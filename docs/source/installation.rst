.. _installation:

Installation
============

To install PyEDDL, you need to install EDDL first. Installation instructions
for EDDL are available in the `EDDL docs
<https://deephealthproject.github.io/eddl/>`_.

Some EDDL components are optional. By default, PyEDDL assumes that EDDL has
been installed with ONNX support. Also, EDDL can be compiled for CPU only or
with GPU support. You can build PyEDDL in both cases, with some differences
that will be explained further ahead.

Each PyEDDL version depends on a specific EDDL version. If you are installing
from the GitHub repo, the correct version of EDDL is available from the
submodule (``third_party`` dir).

The following sections contain instructions to install PyEDDL from scratch,
EDDL installation included. We use Ubuntu Linux as an example platform.


Install EDDL dependencies
-------------------------

EDDL needs the development versions of the zlib and Eigen3 libraries. To
enable ONNX I/O, you also need to install Google protobuf. On Ubuntu, for
instance, install the following with APT: ``build-essential, zlib1g-dev,
libeigen3-dev, wget, ca-certificates``. Install protobuf from source::

    wget -q https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protobuf-all-3.11.4.tar.gz
    tar xf protobuf-all-3.11.4.tar.gz
    cd protobuf-3.11.4
    ./configure
    make -j$(nproc)
    make install


Install EDDL
------------

Clone the PyEDDL GitHub repo::

    git clone --recurse-submodules https://github.com/deephealthproject/pyeddl.git
    cd pyeddl

Install EDDL::

    pushd third_party/eddl
    mkdir build
    pushd build
    cmake -D BUILD_SHARED_LIB=ON -D BUILD_PROTOBUF=ON -D BUILD_TESTS=OFF
    make -j$(nproc)
    make install
    popd
    popd


Install PyEDDL
--------------

You need the development version of python3 and pip. On Ubuntu, install
``python3-dev`` and ``python3-pip``.

Install the Python dependencies::

    python3 -m pip install --upgrade setuptools pip numpy pybind11 pytest

The EDDL code includes Eigen headers like in this example: ``#include
<Eigen/Dense>``, e.g., with ``Eigen`` as the root directory. However, Eigen
installations usually have the header rooted at ``eigen3`` (for instance, the
apt installation places them in ``/usr/include/eigen3``). To work around this
you can either add a symlink or set ``CPATH``, e.g.::

    export CPATH="/usr/include/eigen3:${CPATH}"

Install pyeddl::

    python3 setup.py install

Alternatively, in the case of tagged releases, you can also install PyEDDL
with pip. The following table shows the required EDDL version for each PyEDDL
version:

+----------------+--------------+
| PyECVL version | ECVL version |
+================+==============+
| 0.1.0          | 0.2.2        |
+----------------+--------------+
| 0.2.0          | 0.3          |
+----------------+--------------+
| 0.3.0          | 0.3.1        |
+----------------+--------------+
| 0.4.0          | 0.4.2        |
+----------------+--------------+
| 0.5.0          | 0.4.3        |
+----------------+--------------+
| 0.6.0          | 0.4.4        |
+----------------+--------------+

To install, run::

  python3 -m pip install pyeddl


Disabling unwanted modules
--------------------------

By default, PyEDDL assumes a complete EDDL installation, including optional
modules, and builds bindings for all of them. You can disable support for
specific modules via environment variables. For instance, suppose you
installed EDDL without protobuf support: by default, PyEDDL will try to build
the bindings for protobuf-specific EDDL tools (ONNX support). To avoid this,
set the ``EDDL_WITH_PROTOBUF`` environment variable to ``OFF`` (or ``FALSE``)
before building PyEDDL.


EDDL installed in an arbitrary directory
----------------------------------------

The above installation instructions assume installation in standard system
paths (such as ``/usr/local/include``, ``/usr/local/lib``). However, EDDL can
be installed in an arbitrary directory, for instance::

    cd third_party/eddl
    mkdir build
    cd build
    cmake -D BUILD_SHARED_LIB=ON -D BUILD_PROTOBUF=ON -DCMAKE_INSTALL_PREFIX=/home/myuser/eddl ..
    make
    make install

You can tell the PyEDDL setup script about this via the EDDL_DIR environment
variable::

    export EDDL_DIR=/home/myuser/eddl
    python3 setup.py install

In this way, ``setup.py`` will look for additional include files in
``/home/myuser/eddl/include`` and for additional libraries in
``/home/myuser/eddl/lib``.
