.. _tensor:

:mod:`pyeddl.tensor` --- Tensor
===============================

.. currentmodule:: pyeddl.tensor

.. autodata:: DEV_CPU

.. autodata:: DEV_GPU

.. autoclass:: Tensor

  **=== Creation Methods ===**

  .. automethod:: fromarray

  .. automethod:: Tensor.zeros

  .. automethod:: Tensor.ones

  .. automethod:: Tensor.full

  .. automethod:: Tensor.arange

  .. automethod:: Tensor.range

  .. automethod:: Tensor.linspace

  .. automethod:: Tensor.logspace

  .. automethod:: Tensor.eye

  .. automethod:: Tensor.randn

  **=== Data Copying Methods ===**

  .. automethod:: Tensor.toCPU

  .. automethod:: Tensor.toGPU

  .. automethod:: Tensor.clone

  .. automethod:: Tensor.select

  .. automethod:: Tensor.copy

  **=== Serialization Methods ===**

  .. automethod:: Tensor.load

  .. automethod:: Tensor.save

  **=== Math Methods ===**

  .. automethod:: Tensor.abs_

  .. automethod:: Tensor.abs

  .. automethod:: Tensor.acos_

  .. automethod:: Tensor.acos

  .. automethod:: Tensor.add_

  .. automethod:: Tensor.add

  .. automethod:: Tensor.asin_

  .. automethod:: Tensor.asin

  .. automethod:: Tensor.atan_

  .. automethod:: Tensor.atan

  .. automethod:: Tensor.ceil_

  .. automethod:: Tensor.ceil

  .. automethod:: Tensor.clamp_

  .. automethod:: Tensor.clamp

  .. automethod:: Tensor.clampmax_

  .. automethod:: Tensor.clampmax

  .. automethod:: Tensor.clampmin_

  .. automethod:: Tensor.clampmin

  .. automethod:: Tensor.cos_

  .. automethod:: Tensor.cos

  .. automethod:: Tensor.cosh_

  .. automethod:: Tensor.cosh

  .. automethod:: Tensor.div_

  .. automethod:: Tensor.div

  .. automethod:: Tensor.exp_

  .. automethod:: Tensor.exp

  .. automethod:: Tensor.floor_

  .. automethod:: Tensor.floor

  .. automethod:: Tensor.log_

  .. automethod:: Tensor.log

  .. automethod:: Tensor.log2_

  .. automethod:: Tensor.log2

  .. automethod:: Tensor.log10_

  .. automethod:: Tensor.log10

  .. automethod:: Tensor.logn_

  .. automethod:: Tensor.logn

  .. automethod:: Tensor.max

  .. automethod:: Tensor.min

  .. automethod:: Tensor.mod_

  .. automethod:: Tensor.mod

  .. automethod:: Tensor.mult_

  .. automethod:: Tensor.mult

  .. automethod:: Tensor.mult2D

  .. automethod:: Tensor.neg_

  .. automethod:: Tensor.neg

  .. automethod:: Tensor.normalize_

  .. automethod:: Tensor.normalize

  .. automethod:: Tensor.reciprocal_

  .. automethod:: Tensor.reciprocal

  .. automethod:: Tensor.round_

  .. automethod:: Tensor.round

  .. automethod:: Tensor.rsqrt_

  .. automethod:: Tensor.rsqrt

  .. automethod:: Tensor.sigmoid_

  .. automethod:: Tensor.sigmoid

  .. automethod:: Tensor.sign_

  .. automethod:: Tensor.sign

  .. automethod:: Tensor.sin_

  .. automethod:: Tensor.sin

  .. automethod:: Tensor.sinh_

  .. automethod:: Tensor.sinh

  .. automethod:: Tensor.sqr_

  .. automethod:: Tensor.sqr

  .. automethod:: Tensor.sqrt_

  .. automethod:: Tensor.sqrt

  .. automethod:: Tensor.sub_

  .. automethod:: Tensor.sub

  .. automethod:: Tensor.tan_

  .. automethod:: Tensor.tan

  .. automethod:: Tensor.tanh_

  .. automethod:: Tensor.tanh

  .. automethod:: Tensor.trunc_

  .. automethod:: Tensor.trunc

  **=== Transformations ===**

  .. automethod:: Tensor.scale

  **=== Other Methods ===**

  .. automethod:: Tensor.fill_

  .. automethod:: Tensor.reshape_

  .. automethod:: Tensor.getdata

  .. automethod:: Tensor.print

  .. automethod:: Tensor.info

  .. automethod:: Tensor.getShape

  .. automethod:: Tensor.onehot
