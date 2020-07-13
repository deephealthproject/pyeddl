.. _eddl:

:mod:`pyeddl.eddl` --- eddl API
===============================


Model
-----


Creation
^^^^^^^^

.. autofunction:: pyeddl.eddl.Model

.. autofunction:: pyeddl.eddl.build


Computing services
^^^^^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.toGPU

.. autofunction:: pyeddl.eddl.toCPU

.. autofunction:: pyeddl.eddl.CS_CPU

.. autofunction:: pyeddl.eddl.CS_GPU

.. autofunction:: pyeddl.eddl.CS_FGPA

.. autofunction:: pyeddl.eddl.CS_COMPSS


Info and logs
^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.setlogfile

.. autofunction:: pyeddl.eddl.summary

.. autofunction:: pyeddl.eddl.plot


Serialization
^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.load

.. autofunction:: pyeddl.eddl.save


Optimizers
^^^^^^^^^^

.. autofunction:: pyeddl.eddl.setlr

.. autofunction:: pyeddl.eddl.adadelta

.. autofunction:: pyeddl.eddl.adam

.. autofunction:: pyeddl.eddl.adagrad

.. autofunction:: pyeddl.eddl.adamax

.. autofunction:: pyeddl.eddl.nadam

.. autofunction:: pyeddl.eddl.rmsprop

.. autofunction:: pyeddl.eddl.sgd


Training and evaluation: coarse methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.fit

.. autofunction:: pyeddl.eddl.evaluate


Training and evaluation: finer methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.random_indices

.. autofunction:: pyeddl.eddl.train_batch

.. autofunction:: pyeddl.eddl.eval_batch

.. autofunction:: pyeddl.eddl.next_batch


Training and evaluation: finest methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.set_mode

.. autofunction:: pyeddl.eddl.reset_loss

.. autofunction:: pyeddl.eddl.forward

.. autofunction:: pyeddl.eddl.zeroGrads

.. autofunction:: pyeddl.eddl.backward

.. autofunction:: pyeddl.eddl.update

.. autofunction:: pyeddl.eddl.print_loss


Constraints
^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.clamp

.. autofunction:: pyeddl.eddl.compute_loss


Losses and metrics
^^^^^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.compute_metric

.. autofunction:: pyeddl.eddl.getLoss

.. autofunction:: pyeddl.eddl.newloss

.. autofunction:: pyeddl.eddl.getMetric

.. autofunction:: pyeddl.eddl.detach


Layers
------


Core layers
^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.Activation

.. autofunction:: pyeddl.eddl.Softmax

.. autofunction:: pyeddl.eddl.Sigmoid

.. autofunction:: pyeddl.eddl.HardSigmoid

.. autofunction:: pyeddl.eddl.ReLu

.. autofunction:: pyeddl.eddl.ThresholdedReLu

.. autofunction:: pyeddl.eddl.LeakyReLu

.. autofunction:: pyeddl.eddl.Elu

.. autofunction:: pyeddl.eddl.Selu

.. autofunction:: pyeddl.eddl.Exponential

.. autofunction:: pyeddl.eddl.Softplus

.. autofunction:: pyeddl.eddl.Softsign

.. autofunction:: pyeddl.eddl.Linear

.. autofunction:: pyeddl.eddl.Tanh

.. autofunction:: pyeddl.eddl.Conv

.. autofunction:: pyeddl.eddl.Dense

.. autofunction:: pyeddl.eddl.Dropout

.. autofunction:: pyeddl.eddl.Input

.. autofunction:: pyeddl.eddl.UpSampling

.. autofunction:: pyeddl.eddl.Reshape

.. autofunction:: pyeddl.eddl.Flatten

.. autofunction:: pyeddl.eddl.ConvT

.. autofunction:: pyeddl.eddl.Embedding

.. autofunction:: pyeddl.eddl.Transpose


Transformation layers
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.Crop

.. autofunction:: pyeddl.eddl.CenteredCrop

.. autofunction:: pyeddl.eddl.CropScale

.. autofunction:: pyeddl.eddl.Cutout

.. autofunction:: pyeddl.eddl.Flip

.. autofunction:: pyeddl.eddl.HorizontalFlip

.. autofunction:: pyeddl.eddl.Rotate

.. autofunction:: pyeddl.eddl.Scale

.. autofunction:: pyeddl.eddl.Shift

.. autofunction:: pyeddl.eddl.VerticalFlip


Data augmentation layers
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.RandomCrop

.. autofunction:: pyeddl.eddl.RandomCropScale

.. autofunction:: pyeddl.eddl.RandomCutout

.. autofunction:: pyeddl.eddl.RandomFlip

.. autofunction:: pyeddl.eddl.RandomHorizontalFlip

.. autofunction:: pyeddl.eddl.RandomRotation

.. autofunction:: pyeddl.eddl.RandomScale

.. autofunction:: pyeddl.eddl.RandomShift

.. autofunction:: pyeddl.eddl.RandomVerticalFlip


Merge layers
^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.Add

.. autofunction:: pyeddl.eddl.Average

.. autofunction:: pyeddl.eddl.Concat

.. autofunction:: pyeddl.eddl.MatMul

.. autofunction:: pyeddl.eddl.Maximum

.. autofunction:: pyeddl.eddl.Minimum

.. autofunction:: pyeddl.eddl.Subtract


Noise layers
^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.GaussianNoise


Normalization layers
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.BatchNormalization

.. autofunction:: pyeddl.eddl.LayerNormalization

.. autofunction:: pyeddl.eddl.GroupNormalization

.. autofunction:: pyeddl.eddl.Norm

.. autofunction:: pyeddl.eddl.NormMax

.. autofunction:: pyeddl.eddl.NormMinMax


Operator layers
^^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.Abs

.. autofunction:: pyeddl.eddl.Diff

.. autofunction:: pyeddl.eddl.Div

.. autofunction:: pyeddl.eddl.Exp

.. autofunction:: pyeddl.eddl.Log

.. autofunction:: pyeddl.eddl.Log2

.. autofunction:: pyeddl.eddl.Log10

.. autofunction:: pyeddl.eddl.Mult

.. autofunction:: pyeddl.eddl.Pow

.. autofunction:: pyeddl.eddl.Sqrt

.. autofunction:: pyeddl.eddl.Sum

.. autofunction:: pyeddl.eddl.Select

.. autofunction:: pyeddl.eddl.Permute


Reduction layers
^^^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.ReduceMean

.. autofunction:: pyeddl.eddl.ReduceVar

.. autofunction:: pyeddl.eddl.ReduceSum

.. autofunction:: pyeddl.eddl.ReduceMax

.. autofunction:: pyeddl.eddl.ReduceMin


Generator layers
^^^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.GaussGenerator

.. autofunction:: pyeddl.eddl.UniformGenerator


Pooling layers
^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.AveragePool

.. autofunction:: pyeddl.eddl.GlobalMaxPool

.. autofunction:: pyeddl.eddl.GlobalAveragePool

.. autofunction:: pyeddl.eddl.MaxPool


Recurrent layers
^^^^^^^^^^^^^^^^

.. autofunction:: pyeddl.eddl.RNN

.. autofunction:: pyeddl.eddl.LSTM


Utilities
^^^^^^^^^

.. autofunction:: pyeddl.eddl.set_trainable

.. autofunction:: pyeddl.eddl.getOut


Initializers
------------

.. autofunction:: pyeddl.eddl.GlorotNormal

.. autofunction:: pyeddl.eddl.GlorotUniform

.. autofunction:: pyeddl.eddl.RandomNormal

.. autofunction:: pyeddl.eddl.RandomUniform

.. autofunction:: pyeddl.eddl.Constant


Regularizers
------------

.. autofunction:: pyeddl.eddl.L2

.. autofunction:: pyeddl.eddl.L1

.. autofunction:: pyeddl.eddl.L1L2


Datasets
--------

.. autofunction:: pyeddl.eddl.exist

.. autofunction:: pyeddl.eddl.download_mnist

.. autofunction:: pyeddl.eddl.download_cifar10

.. autofunction:: pyeddl.eddl.download_drive


ONNX support
------------

.. autofunction:: pyeddl.eddl.save_net_to_onnx_file

.. autofunction:: pyeddl.eddl.import_net_from_onnx_file

.. autofunction:: pyeddl.eddl.serialize_net_to_onnx_string

.. autofunction:: pyeddl.eddl.import_net_from_onnx_string
