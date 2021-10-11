.. _installation:

Installation
============


Conda packages
--------------

`Conda <https://docs.conda.io/en/latest/>`_ packages for PyEDDL come in three
flavors:

* ``pyeddl-cpu``: CPU-only
* ``pyeddl-gpu``: GPU-enabled
* ``pyeddl-cudnn``: GPU-enabled, with cuDNN support

Choose one of the above, then install it as follows::

  conda install -c dhealth -c bioconda -c conda-forge pyeddl-cpu
  conda install -c dhealth -c bioconda -c conda-forge pyeddl-gpu
  conda install -c dhealth -c bioconda -c conda-forge pyeddl-cudnn

Each PyEDDL package installs the corresponding EDDL one as a dependency, as
well as any other requirement, so there's no need to separately install EDDL
or PyEDDL / EDDL when installing via Conda.

.. note::

   The highest version available from Conda might be less than the latest
   version on PyPI for recent releases.

Head over to https://github.com/deephealthproject/conda_builds for further
information.


Docker images
-------------

If you don't want to install on bare metal, the `DeepHealth Docker images
<https://github.com/deephealthproject/docker-libs>`_ provide ready-to-use
containers for the DeepHealth components, including PyEDDL. Refer to the above
link for further information.


Installing from source
----------------------

To install PyEDDL from source, you need to install EDDL first. Installation
instructions for EDDL are available as part of the `EDDL docs
<https://deephealthproject.github.io/eddl/>`_, although we also provide
some hints here. Please refer to the EDDL docs if you need more details.

Some EDDL components are optional. By default, PyEDDL assumes that EDDL has
been installed with ONNX support. Also, EDDL can be compiled for CPU only or
with GPU/CUDNN support. You can build PyEDDL in both cases, with some
differences that will be highlighted further ahead.

Each PyEDDL version depends on a specific EDDL version. If you are installing
from the GitHub repo, the correct version of EDDL is available from the
submodule (``third_party`` dir).

The following sections contain instructions to install PyEDDL from scratch,
EDDL installation included. We use Ubuntu Linux as an example platform.

Install EDDL dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^

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

If you want to compile EDDL for **GPU/CUDNN**, you also need the CUDA toolkit
(``nvidia-cuda-toolkit``). In this case, `you also need a recent version of
Eigen3
<https://devtalk.nvidia.com/default/topic/1026622/nvcc-can-t-compile-code-that-uses-eigen>`_.
In the CUDNN case, you also need to install cuDNN.

.. note::

   PyEDDL is part of the `DeepHealth ecosystem
   <https://github.com/deephealthproject>`_. If you plan to use PyEDDL with
   other DeepHealth components, you should make sure that your environment
   satisfies the requirements of all such components. In particular, `ECVL
   <https://github.com/deephealthproject/ecvl>`_ needs a compiler with support
   for C++ 17. Being a set of bindings for ECVL,
   `PyECVL <https://github.com/deephealthproject/pyecvl>`_ also needs a C++ 17
   compiler. Since all components need to be built with the same compiler to
   work together, if you want to use PyEDDL together with PyECVL, you need to
   compile everything with the same C++ 17 compiler. See the `PyECVL docs
   <https://deephealthproject.github.io/pyecvl>`_ for an example of PyECVL +
   PyEDDL installation.


Install EDDL
^^^^^^^^^^^^

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

To compile for **GPU**, add ``-D BUILD_TARGET=GPU`` to the cmake flags.
To compile for GPU with CUDNN support, add  ``-D BUILD_TARGET=CUDNN`` instead.

.. note::

    EDDL versions 0.3 and 0.3.1 are affected by an issue that breaks the
    building of the shared library. To work around this, you can patch the
    EDDL code with ``eddl_0.3.patch`` (at the top level in the PyEDDL git
    repository).


Install PyEDDL
^^^^^^^^^^^^^^

You need the development version of python3 and pip. On Ubuntu, install
``python3-dev`` and ``python3-pip``.

Install the Python dependencies. Note that, at the moment, the pybind11
version needs to be less than 2.6::

    python3 -m pip install --upgrade setuptools pip
    python3 -m pip install --upgrade numpy 'pybind11<2.6' pytest

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
| PyEDDL version | EDDL version |
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
| 0.7.*          | 0.5.4a       |
+----------------+--------------+
| 0.8.*          | 0.6.0        |
+----------------+--------------+
| 0.9.*          | 0.7.1        |
+----------------+--------------+
| 0.10.*         | 0.8a         |
+----------------+--------------+
| 0.11.*         | 0.8.1a       |
+----------------+--------------+
| 0.12.*         | 0.8.3a       |
+----------------+--------------+
| 0.13.*         | 0.9.1b       |
+----------------+--------------+
| 0.14.*         | 0.9.2b       |
+----------------+--------------+
| 1.0.0          | 1.0.2a       |
+----------------+--------------+
| 1.1.0          | 1.0.3b       |
+----------------+--------------+

To install, run::

  python3 -m pip install pyeddl

If EDDL was compiled for GPU/CUDNN, you need to export the ``EDDL_WITH_CUDA``
environment variable **before installing PyEDDL** so that ``setup.py`` will
also link the ``cudart``, ``cublas`` and ``curand`` libraries. These will be
expected in "standard" system locations, so you might need to create symlinks
depending on your CUDA toolkit installation. For instance::

    export EDDL_WITH_CUDA="true"
    ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so /usr/lib/
    ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcurand.so /usr/lib/
    ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcublas.so /usr/lib/


Disabling unwanted modules
^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, PyEDDL assumes a complete EDDL installation, including optional
modules, and builds bindings for all of them. You can disable support for
specific modules via environment variables. For instance, suppose you
installed EDDL without protobuf support: by default, PyEDDL will try to build
the bindings for protobuf-specific EDDL tools (ONNX support). To avoid this,
set the ``EDDL_WITH_PROTOBUF`` environment variable to ``OFF`` (or ``FALSE``)
before building PyEDDL.


EDDL installed in an arbitrary directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
