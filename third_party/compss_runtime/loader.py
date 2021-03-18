from typing import Text

import numpy as np

from eddl_array import array


def load_npy(filename: Text, num_workers: int):
    x = np.load(file=filename, mmap_mode='r')
    block_size = int(x.shape[0] / num_workers)
    return array(x, block_size)
