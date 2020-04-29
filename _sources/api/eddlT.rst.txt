.. _eddlT:

:mod:`pyeddl.eddlT` --- eddlT API
=================================

.. autodata:: pyeddl.eddlT.DEV_CPU

.. autodata:: pyeddl.eddlT.DEV_GPU


Creation ops
------------

.. autofunction:: pyeddl.eddlT.create

.. autofunction:: pyeddl.eddlT.zeros

.. autofunction:: pyeddl.eddlT.ones

.. autofunction:: pyeddl.eddlT.full

.. autofunction:: pyeddl.eddlT.arange

.. autofunction:: pyeddl.eddlT.range

.. autofunction:: pyeddl.eddlT.linspace

.. autofunction:: pyeddl.eddlT.logspace

.. autofunction:: pyeddl.eddlT.eye

.. autofunction:: pyeddl.eddlT.randn


Copy data
---------

.. autofunction:: pyeddl.eddlT.toCPU_

.. autofunction:: pyeddl.eddlT.toGPU_

.. autofunction:: pyeddl.eddlT.toCPU

.. autofunction:: pyeddl.eddlT.toGPU

.. autofunction:: pyeddl.eddlT.clone

.. autofunction:: pyeddl.eddlT.select

.. autofunction:: pyeddl.eddlT.copyTensor


Core inplace
------------

.. autofunction:: pyeddl.eddlT.fill_

.. autofunction:: pyeddl.eddlT.set_

.. autofunction:: pyeddl.eddlT.reshape_


Get data
--------

.. autofunction:: pyeddl.eddlT.getdata


Serialization
-------------

.. autofunction:: pyeddl.eddlT.load

.. autofunction:: pyeddl.eddlT.save


Math ops
--------

.. autofunction:: pyeddl.eddlT.abs_

.. autofunction:: pyeddl.eddlT.abs

.. autofunction:: pyeddl.eddlT.acos_

.. autofunction:: pyeddl.eddlT.acos

.. autofunction:: pyeddl.eddlT.add_

.. autofunction:: pyeddl.eddlT.add

.. autofunction:: pyeddl.eddlT.asin_

.. autofunction:: pyeddl.eddlT.asin

.. autofunction:: pyeddl.eddlT.atan_

.. autofunction:: pyeddl.eddlT.atan

.. autofunction:: pyeddl.eddlT.ceil_

.. autofunction:: pyeddl.eddlT.ceil

.. autofunction:: pyeddl.eddlT.clamp_

.. autofunction:: pyeddl.eddlT.clamp

.. autofunction:: pyeddl.eddlT.clampmax_

.. autofunction:: pyeddl.eddlT.clampmax

.. autofunction:: pyeddl.eddlT.clampmin_

.. autofunction:: pyeddl.eddlT.clampmin

.. autofunction:: pyeddl.eddlT.cos_

.. autofunction:: pyeddl.eddlT.cos

.. autofunction:: pyeddl.eddlT.cosh_

.. autofunction:: pyeddl.eddlT.cosh

.. autofunction:: pyeddl.eddlT.div_

.. autofunction:: pyeddl.eddlT.div

.. autofunction:: pyeddl.eddlT.exp_

.. autofunction:: pyeddl.eddlT.exp

.. autofunction:: pyeddl.eddlT.floor_

.. autofunction:: pyeddl.eddlT.floor

.. autofunction:: pyeddl.eddlT.inc_

.. autofunction:: pyeddl.eddlT.log_

.. autofunction:: pyeddl.eddlT.log

.. autofunction:: pyeddl.eddlT.log2_

.. autofunction:: pyeddl.eddlT.log2

.. autofunction:: pyeddl.eddlT.log10_

.. autofunction:: pyeddl.eddlT.log10

.. autofunction:: pyeddl.eddlT.logn_

.. autofunction:: pyeddl.eddlT.logn

.. autofunction:: pyeddl.eddlT.max

.. autofunction:: pyeddl.eddlT.min

.. autofunction:: pyeddl.eddlT.mod_

.. autofunction:: pyeddl.eddlT.mod

.. autofunction:: pyeddl.eddlT.mult_

.. autofunction:: pyeddl.eddlT.mult

.. autofunction:: pyeddl.eddlT.mult2D

.. autofunction:: pyeddl.eddlT.neg_

.. autofunction:: pyeddl.eddlT.neg

.. autofunction:: pyeddl.eddlT.normalize_

.. autofunction:: pyeddl.eddlT.normalize

.. autofunction:: pyeddl.eddlT.reciprocal_

.. autofunction:: pyeddl.eddlT.reciprocal

.. autofunction:: pyeddl.eddlT.round_

.. autofunction:: pyeddl.eddlT.round

.. autofunction:: pyeddl.eddlT.rsqrt_

.. autofunction:: pyeddl.eddlT.rsqrt

.. autofunction:: pyeddl.eddlT.sigmoid_

.. autofunction:: pyeddl.eddlT.sigmoid

.. autofunction:: pyeddl.eddlT.sign_

.. autofunction:: pyeddl.eddlT.sign

.. autofunction:: pyeddl.eddlT.sin_

.. autofunction:: pyeddl.eddlT.sin

.. autofunction:: pyeddl.eddlT.sinh_

.. autofunction:: pyeddl.eddlT.sinh

.. autofunction:: pyeddl.eddlT.sqr_

.. autofunction:: pyeddl.eddlT.sqr

.. autofunction:: pyeddl.eddlT.sqrt_

.. autofunction:: pyeddl.eddlT.sqrt

.. autofunction:: pyeddl.eddlT.sub_

.. autofunction:: pyeddl.eddlT.sub

.. autofunction:: pyeddl.eddlT.tan_

.. autofunction:: pyeddl.eddlT.tan

.. autofunction:: pyeddl.eddlT.tanh_

.. autofunction:: pyeddl.eddlT.tanh

.. autofunction:: pyeddl.eddlT.trunc_

.. autofunction:: pyeddl.eddlT.trunc


Reduction
---------

.. autofunction:: pyeddl.eddlT.reduce_mean

.. autofunction:: pyeddl.eddlT.reduce_variance


Other functions
---------------

.. autofunction:: pyeddl.eddlT.print

.. autofunction:: pyeddl.eddlT.info

.. autofunction:: pyeddl.eddlT.getShape
