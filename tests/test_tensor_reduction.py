import numpy as np
from pyeddl._core import Tensor, ReduceDescriptor, reduction
from pyeddl.api import DEV_CPU


def test_reduction():
    t = Tensor([10, 5], DEV_CPU)
    axis = 0
    rd = ReduceDescriptor(t, [axis], "mean", False)
    t.rand_uniform(1.0)
    a = np.array(t, copy=False)
    reduction(rd)
    b = np.array(rd.O, copy=False)
    assert np.allclose(np.mean(a, axis), b)
