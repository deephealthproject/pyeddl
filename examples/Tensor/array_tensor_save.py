import numpy as np
import pyeddl._core.eddlT as eddlT

a = np.arange(6).reshape([2, 3]).astype(np.float32)
print(a)
t = eddlT.create(a)
eddlT.save(t, "a.bin", "bin")
t1 = eddlT.load("a.bin", "bin")
a1 = eddlT.getdata(t1)
print()
print(a1)
