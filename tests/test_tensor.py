import unittest

import numpy as np

# from pyeddl.layers import Tensor
from pyeddl import _C

t_array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])


class TestTensors(unittest.TestCase):

    def test_device(self):
        for dev in [_C.DEV_CPU]:
            t = _C.Tensor(t_array.shape, dev)
            self.assertEqual(dev, t.device)

    def test_ndim(self):
        t = _C.Tensor(t_array.shape, _C.DEV_CPU)
        self.assertEqual(len(t_array.shape), t.ndim)

    def test_size(self):
        t = _C.Tensor(t_array.shape, _C.DEV_CPU)
        self.assertEqual(t_array.size, t.size)

    def test_shape(self):
        t_shape = t_array.shape
        t = _C.Tensor(t_shape, _C.DEV_CPU)
        c_shape = t.shape

        # Check for dim
        self.assertEqual(len(t_shape), len(c_shape))
        for ts, cs in zip(t_shape, c_shape):
            self.assertEqual(ts, cs)

    def test_tensor_from_npy(self):
        new_arr = np.asarray(t_array, dtype=np.float32)
        t = _C.tensor_from_npy(new_arr, _C.DEV_CPU)

        # Get data
        res = _C.tensor_getdata(t)

        # Check if there are the same
        self.assertTrue(np.array_equal(new_arr, res))

        # Modify original array
        new_arr *= 2.0

        # Check if there are NOT the same
        self.assertFalse(np.array_equal(new_arr, res))


if __name__ == "__main__":
    unittest.main()
