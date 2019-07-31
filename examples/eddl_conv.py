from pyeddl.api import (
    Input, Activation, Dense, Model, sgd, CS_CPU, build, T_load, div, fit,
    evaluate, Reshape, MaxPool, Conv
)
from pyeddl.utils import download_mnist


def main():
    download_mnist()

    epochs = 5
    batch_size = 1000
    num_classes = 10

    in_ = Input([784])
    layer = in_

    layer = Reshape(layer, [1, 28, 28])

    layer = MaxPool(Activation(Conv(layer, 16, [3, 3]), "relu"), [2, 2])
    layer = MaxPool(Activation(Conv(layer, 32, [3, 3]), "relu"), [2, 2])
    layer = MaxPool(Activation(Conv(layer, 64, [3, 3]), "relu"), [2, 2])
    layer = MaxPool(Activation(Conv(layer, 128, [3, 3]), "relu"), [2, 2])

    layer = Reshape(layer, [-1])

    layer = Activation(Dense(layer, 32), "relu")
    out = Activation(Dense(layer, num_classes), "softmax")

    net = Model([in_], [out])
    print(net.summary())

    build(
        net,
        sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        CS_CPU(4)
    )

    X = T_load("trX.bin")
    Y = T_load("trY.bin")

    div(X, 255.0)

    fit(net, [X], [Y], batch_size, epochs)
    evaluate(net, [X], [Y])

    tX = T_load("tsX.bin")
    tY = T_load("tsY.bin")

    div(tX, 255.0)

    evaluate(net, [tX], [tY])


if __name__ == "__main__":
    main()
