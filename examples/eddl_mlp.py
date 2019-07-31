import pyeddl._core as pyeddl
from pyeddl.utils import download_mnist, loss_func, metric_func


num_classes = 10
epochs = 10
batch_size = 1000

t = pyeddl.Tensor([1, 784], 0)
i = pyeddl.LInput(t, "foo", 0)
l = i
l = pyeddl.LActivation(pyeddl.LDense(l, 1024, True, "", 0), "relu", "", 0)
l = pyeddl.LActivation(pyeddl.LDense(l, 1024, True, "", 0), "relu", "", 0)
l = pyeddl.LActivation(pyeddl.LDense(l, 1024, True, "", 0), "relu", "", 0)
o = pyeddl.LActivation(pyeddl.LDense(l, num_classes, True, "", 0), "softmax", "", 0)
n = pyeddl.Net([i], [o])
print(n.summary())

optimizer = pyeddl.SGD(0.01, 0.9)
losses = [loss_func("soft_cross_entropy")]
metrics = [metric_func("categorical_accuracy")]
compserv = pyeddl.CompServ(4, [], [])

n.build(optimizer, losses, metrics, compserv)

download_mnist()

x_train = pyeddl.LTensor("trX.bin")
y_train = pyeddl.LTensor("trY.bin")
x_test = pyeddl.LTensor("tsX.bin")
y_test = pyeddl.LTensor("tsY.bin")

x_train.input.div(255.0)
x_test.input.div(255.0)

n.fit([x_train.input], [y_train.input], batch_size, epochs)
n.evaluate([x_test.input], [y_test.input])
