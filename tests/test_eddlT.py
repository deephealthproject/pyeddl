import numpy as np
import pyeddl._core.eddlT as eddlT


def test_create_getdata():
    R, C = 3, 4
    t = eddlT.create([R, C])
    assert t.shape == [R, C]
    t = eddlT.create([R, C], eddlT.DEV_CPU)
    assert t.shape == [R, C]
    a = np.arange(R * C).reshape(R, C).astype(np.float32)
    t = eddlT.create(a)
    assert t.shape == [R, C]
    b = eddlT.getdata(t)
    assert np.array_equal(b, a)
    # check automatic type conversion
    a = np.arange(R * C).reshape(R, C)
    t = eddlT.create(a)
    assert t.shape == [R, C]
