# Copyright (c) 2019-2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from urllib.request import urlretrieve

import numpy as np
from pyeddl.tensor import Tensor

# Convert array to tensor and save to "bin" format
a = np.arange(6).reshape([2, 3]).astype(np.float32)
print(a)
t = Tensor.fromarray(a)
t.save("./a.bin", "bin")
t1 = Tensor.load("a.bin", "bin")
a1 = t1.getdata()
print(a1)

print()

# Read numpy data and convert to tensors
FNAME = "mnist.npz"
LOC = "https://storage.googleapis.com/tensorflow/tf-keras-datasets"
if not os.path.exists(FNAME):
    fname, _ = urlretrieve("%s/%s" % (LOC, FNAME), FNAME)
    print("Downloaded", fname)
print("loading", FNAME)
with np.load(FNAME) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
t_x_train = Tensor.fromarray(x_train.astype(np.float32))
t_y_train = Tensor.fromarray(y_train.astype(np.float32))
t_x_test = Tensor.fromarray(x_test.astype(np.float32))
t_y_test = Tensor.fromarray(y_test.astype(np.float32))
print(x_train.shape, x_train.dtype)
t_x_train.info()
