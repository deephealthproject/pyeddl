# Copyright (c) 2019-2020, CRS4
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
from urllib.request import urlretrieve

import numpy as np
import pyeddl.eddlT as eddlT

# Convert array to tensor and save to "bin" format
a = np.arange(6).reshape([2, 3]).astype(np.float32)
print(a)
t = eddlT.create(a)
eddlT.save(t, "./a.bin", "bin")
t1 = eddlT.load("a.bin", "bin")
a1 = eddlT.getdata(t1)
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
t_x_train = eddlT.create(x_train.astype(np.float32))
t_y_train = eddlT.create(y_train.astype(np.float32))
t_x_test = eddlT.create(x_test.astype(np.float32))
t_y_test = eddlT.create(y_test.astype(np.float32))
print(x_train.shape, x_train.dtype)
eddlT.info(t_x_train)
