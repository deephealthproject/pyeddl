import numpy as np
from pyeddl._core import Tensor
import pyeddl._core.eddlT as eddlT

a = np.arange(6).reshape([2, 3]).astype(np.float32)
print(a)
t = Tensor(a)
eddlT.save(t, "a.bin", "bin")
t1 = eddlT.load("a.bin", "bin")
a1 = np.array(t1, copy=False)
print()
print(a1)
