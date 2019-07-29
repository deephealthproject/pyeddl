import pyeddl._core as pyeddl
from eddl_utils import download_mnist, loss_func, metric_func

epochs = 5
batch_size = 1000
num_classes = 10

i = pyeddl.LInput(pyeddl.Tensor([1, 784], 0), "foo", 0)
l = i
l = pyeddl.LReshape(l, [1, 1, 28, 28], "reshape1", 0)

pd = pyeddl.PoolDescriptor([2, 2], [2, 2], "none")

l = pyeddl.LMaxPool(pyeddl.LActivation(
    pyeddl.LConv(l, 16, [3, 3], [1, 1], "same", 1, [1, 1], True, "", 0),
    "relu", "", 0), pd, "mx1", 0)
l = pyeddl.LMaxPool(pyeddl.LActivation(
    pyeddl.LConv(l, 32, [3, 3], [1, 1], "same", 1, [1, 1], True, "", 0),
    "relu", "", 0), pd, "mx2", 0)
l = pyeddl.LMaxPool(pyeddl.LActivation(
    pyeddl.LConv(l, 64, [3, 3], [1, 1], "same", 1, [1, 1], True, "", 0),
    "relu", "", 0), pd, "mx3", 0)
l = pyeddl.LMaxPool(pyeddl.LActivation(
    pyeddl.LConv(l, 128, [3, 3], [1, 1], "same", 1, [1, 1], True, "", 0),
    "relu", "", 0), pd, "mx4", 0)

l = pyeddl.LReshape(l, [1, -1], "reshape2", 0)

l = pyeddl.LActivation(pyeddl.LDense(l, 32, True, "", 0), "relu", "", 0)
o = pyeddl.LActivation(pyeddl.LDense(l, num_classes, True, "", 0),
                       "softmax", "", 0)

n = pyeddl.Net([i], [o])
print(n.summary())

optimizer = pyeddl.SGD(0.01, 0.9)
losses = [loss_func("soft_cross_entropy")]
metrics = [metric_func("categorical_accuracy")]
compserv = pyeddl.CompServ(4, [], [])

n.build(optimizer, losses, metrics, compserv)

download_mnist()

x = pyeddl.LTensor("trX.bin")
y = pyeddl.LTensor("trY.bin")

x.input.div(255.0)

n.fit([x.input], [y.input], batch_size, epochs)
n.evaluate([x.input], [y.input])

tx = pyeddl.LTensor("tsX.bin")
ty = pyeddl.LTensor("tsY.bin")

tx.input.div(255.0)

n.evaluate([tx.input], [ty.input])
