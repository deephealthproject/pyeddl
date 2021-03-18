from typing import MutableSequence

import numpy as np
from dislib import utils
from dislib.data.array import Array
from pycompss.api.parameter import *
from pycompss.api.task import task
from pyeddl.tensor import Tensor as eddlT
from scipy.sparse import issparse
from pycompss.api.constraint import constraint


# noinspection PyProtectedMember
#@task(tensor={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
@constraint(computing_units="${OMP_NUM_THREADS}")
@task(returns=np.array)
def _apply_to_tensor(func, tensor, *args, **kwargs):
    tensor = to_tensor(tensor)
    func(tensor, *args, **kwargs)
    return eddlT.getdata(tensor)


# noinspection PyProtectedMember
def apply(func, x, *args, **kwargs):
    out_blocks = []
    for block in x._iterator():
        out = _apply_to_tensor(func, block._blocks, *args, **kwargs)
        out_blocks.append([out])
    return Array(blocks=out_blocks,
                 top_left_shape=x._top_left_shape,
                 reg_shape=x._reg_shape,
                 shape=x.shape,
                 sparse=False)


# noinspection PyProtectedMember
def array(x, block_size):
    if not isinstance(x, np.ndarray):
        x = eddlT.getdata(x)
    if not isinstance(block_size, MutableSequence):
        block_size = (block_size, *x.shape[1:])
    blocks = []
    for i in range(0, x.shape[0], block_size[0]):
        row = x[i: i + block_size[0], :]
        blocks.append([row])
    sparse = issparse(x)
    arr = Array(blocks=blocks,
                top_left_shape=block_size,
                reg_shape=block_size,
                shape=x.shape,
                sparse=sparse)
    print("DisLib array: " + str(arr))
    return arr


# noinspection PyProtectedMember
def paired_partition(x, y):
    for block_x, block_y in utils.base._paired_partition(x, y):
        yield block_x._blocks, block_y._blocks


def to_tensor(tensor):
    tensor = Array._merge_blocks(tensor)
    return eddlT.fromarray(tensor)
