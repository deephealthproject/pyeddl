import numpy as np
from pyeddl._core import Tensor, PoolDescriptor, MPool2D
from pyeddl.api import DEV_CPU


def test_MPool2D():
    ref = [12.0, 20.0, 30.0, 0.0,
           8.0, 12.0, 2.0, 0.0,
           34.0, 70.0, 37.0, 4.0,
           112.0, 100.0, 25.0, 12.0]
    sol = [20.0, 30.0,
           112.0, 37.0]
    mpool_ref = np.array(ref, dtype=np.float32).reshape(1, 1, 4, 4)
    mpool_sol = np.array(sol, dtype=np.float32).reshape(1, 1, 2, 2)
    t = Tensor(mpool_ref, DEV_CPU)
    pd = PoolDescriptor([2, 2], [2, 2], "none")
    pd.build(t)
    pd.indX = Tensor(pd.O.getShape(), DEV_CPU)
    pd.indY = Tensor(pd.O.getShape(), DEV_CPU)
    MPool2D(pd)
    a = np.array(pd.O, copy=False)
    assert np.array_equal(a, mpool_sol)
