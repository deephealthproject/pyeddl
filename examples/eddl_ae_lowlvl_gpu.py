import pyeddl._core as pyeddl
from pyeddl.utils import download_mnist, loss_func, metric_func

epochs = 10
batch_size = 1000

t = pyeddl.Tensor([1, 784], 1000)
i = pyeddl.LInput(t, "foo", 1000)
l = i
l = pyeddl.LActivation(pyeddl.LDense(l, 256, True, "", 1000), "relu", "", 1000)
l = pyeddl.LActivation(pyeddl.LDense(l, 128, True, "", 1000), "relu", "", 1000)
l = pyeddl.LActivation(pyeddl.LDense(l, 64, True, "", 1000), "relu", "", 1000)
l = pyeddl.LActivation(pyeddl.LDense(l, 128, True, "", 1000), "relu", "", 1000)
l = pyeddl.LActivation(pyeddl.LDense(l, 256, True, "", 1000), "relu", "", 1000)
o = pyeddl.LDense(l, 784, True, "", 1000)
n = pyeddl.Net([i], [o])
print(n.summary())

optimizer = pyeddl.SGD(0.01, 0.9)
losses = [loss_func("mean_squared_error")]
metrics = [metric_func("mean_squared_error")]
compserv = pyeddl.CompServ(0, [1], [], 10)

n.build(optimizer, losses, metrics, compserv)

download_mnist()

x_train = pyeddl.LTensor("trX.bin")

x_train.input.div(255.0)

n.fit([x_train.input], [x_train.input], batch_size, epochs)
