import numpy as np

from dislib import utils
from dislib.data.array import Array
from pycompss.api.parameter import *
from pycompss.api.task import task


# noinspection PyProtectedMember
@task(source_x={Type: COLLECTION_IN, Depth: 2},
      source_y={Type: COLLECTION_IN, Depth: 2},
      target_x={Type: COLLECTION_IN, Depth: 2},
      target_y={Type: COLLECTION_IN, Depth: 2},
      out_target_x=INOUT,
      out_target_y=INOUT,
      returns=(np.array, np.array))
def _block_exchange(source_x, source_y, source_indexes,
                    target_x, target_y, target_indexes,
                    out_target_x, out_target_y):
    source_x = Array._merge_blocks(source_x)
    source_y = Array._merge_blocks(source_y)
    target_x = Array._merge_blocks(target_x)
    target_y = Array._merge_blocks(target_y)
    if len(source_indexes) < len(target_indexes):
        out_target_x[target_indexes[len(source_indexes):]] = target_x[target_indexes[len(source_indexes):]]
        out_target_y[target_indexes[len(source_indexes):]] = target_y[target_indexes[len(source_indexes):]]
        target_indexes = target_indexes[:len(source_indexes)]
    out_target_x[target_indexes] = source_x[source_indexes[:len(target_indexes)]]
    out_target_y[target_indexes] = source_y[source_indexes[:len(target_indexes)]]
    source_x[source_indexes[:len(target_indexes)]] = target_x[target_indexes]
    source_y[source_indexes[:len(target_indexes)]] = target_y[target_indexes]
    return source_x, source_y


# noinspection PyProtectedMember
@task(
    x={Type: COLLECTION_IN, Depth: 2},
    y={Type: COLLECTION_IN, Depth: 2},
    returns=(np.array, np.array))
def block_shuffle(x, y):
    x = Array._merge_blocks(x)
    y = Array._merge_blocks(y)
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]


# noinspection PyProtectedMember
@task(
    x={Type: COLLECTION_IN, Depth: 2},
    y={Type: COLLECTION_IN, Depth: 2},
    parameters=IN,
    returns=(np.array, np.array))
def block_shuffle_async(x, y, parameters):
    x = Array._merge_blocks(x)
    y = Array._merge_blocks(y)
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]


# noinspection PyProtectedMember
def _get_paired_block(x, y, block_index):
    row = x._blocks[block_index]
    block_x = Array(blocks=[row],
                    top_left_shape=(x._get_row_shape(block_index)[0], x._top_left_shape[1]),
                    reg_shape=x._reg_shape, shape=x._get_row_shape(block_index),
                    sparse=x._sparse)
    start_idx = 0
    end_idx = x._top_left_shape[0]
    for i in range(block_index + 1):
        start_idx = end_idx
        end_idx = min(end_idx + x._reg_shape[0], x.shape[0])
    block_y = y[start_idx:end_idx]
    return block_x, block_y


# noinspection PyProtectedMember
@task(
    target_x={Type: COLLECTION_IN, Depth: 2},
    target_y={Type: COLLECTION_IN, Depth: 2},
    out_target_x=INOUT,
    out_target_y=INOUT)
def _insert_target_remainder(target_x, target_y, target_indexes,
                             out_target_x, out_target_y):
    target_x = Array._merge_blocks(target_x)
    target_y = Array._merge_blocks(target_y)
    out_target_x[target_indexes] = target_x[target_indexes]
    out_target_y[target_indexes] = target_y[target_indexes]


# noinspection PyProtectedMember
def local_shuffle(x, y):
    out_blocks_x = []
    out_blocks_y = []
    for block_x, block_y in utils.base._paired_partition(x, y):
        out_x, out_y = block_shuffle(block_x._blocks, block_y._blocks)
        out_blocks_x.append([out_x])
        out_blocks_y.append([out_y])
    shuffled_x = Array(blocks=out_blocks_x, top_left_shape=x._top_left_shape, reg_shape=x._reg_shape,
                       shape=x.shape, sparse=False)
    shuffled_y = Array(blocks=out_blocks_y, top_left_shape=y._top_left_shape, reg_shape=y._reg_shape,
                       shape=y.shape, sparse=False)

    return shuffled_x, shuffled_y


def global_shuffle(x, y):
    return utils.shuffle(x, y)


# noinspection PyProtectedMember
def selective_shuffle(x, y, block_index):
    out_blocks_x = []
    out_blocks_y = []
    target_x, target_y = _get_paired_block(x, y, block_index)
    target_elements = target_x.shape[0]
    n_blocks = len(x._blocks) - 1
    elements_per_slice = target_elements // n_blocks
    target_perm = np.random.permutation(target_elements)
    out_target_x = np.empty(target_x.shape)
    out_target_y = np.empty(target_y.shape)
    start_idx = 0
    end_idx = elements_per_slice
    for i, (source_x, source_y) in enumerate(utils.base._paired_partition(x, y)):
        if i == block_index:
            continue
        source_elements = source_x.shape[0]
        source_perm = np.random.permutation(source_elements)

        src_x, src_y = _block_exchange(source_x._blocks, source_y._blocks, source_perm,
                                       target_x._blocks, target_y._blocks, target_perm[start_idx:end_idx],
                                       out_target_x, out_target_y)
        out_blocks_x.append([src_x])
        out_blocks_y.append([src_y])
        start_idx = end_idx
        end_idx = min(end_idx + elements_per_slice, target_elements)
    if (target_elements % n_blocks) != 0:
        _insert_target_remainder(target_x._blocks, target_y._blocks, target_perm[start_idx:end_idx],
                                 out_target_x, out_target_y)
    out_blocks_x.insert(block_index, [out_target_x])
    out_blocks_y.insert(block_index, [out_target_y])
    shuffled_x = Array(blocks=out_blocks_x, top_left_shape=x._top_left_shape, reg_shape=x._reg_shape,
                       shape=x.shape, sparse=False)
    shuffled_y = Array(blocks=out_blocks_y, top_left_shape=y._top_left_shape, reg_shape=y._reg_shape,
                       shape=y.shape, sparse=False)
    return shuffled_x, shuffled_y
