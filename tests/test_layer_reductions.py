import numpy as np
from pyeddl._core import LTensor, LRMean
from pyeddl.api import DEV_CPU


def test_lrmean():
    axis = 0
    lt = LTensor([10, 5], DEV_CPU)
    lt.output.rand_uniform(1.0)
    a = np.array(lt.output, copy=False)
    lrm = LRMean(lt, [axis], True, "LTMean", DEV_CPU)
    lrm.forward()
    b = np.array(lrm.output, copy=False)
    assert np.allclose(np.mean(a, axis), b[0])
