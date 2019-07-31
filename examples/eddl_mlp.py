from pyeddl.api import (
    Input, Activation, Dense, Model, sgd, CS_CPU, build, T_load, div, fit,
    evaluate, GaussianNoise
)
from pyeddl.utils import download_mnist


def main():
    download_mnist()

    epochs = 10
    batch_size = 1000
    num_classes = 10

    in_ = Input([784])
    layer = in_
    layer = GaussianNoise(layer, 0.3)
    layer = Activation(Dense(layer, 1024), "relu")
    layer = Activation(Dense(layer, 1024), "relu")
    layer = Activation(Dense(layer, 1024), "relu")
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

    x_train = T_load("trX.bin")
    y_train = T_load("trY.bin")
    x_test = T_load("tsX.bin")
    y_test = T_load("tsY.bin")

    div(x_train, 255.0)
    div(x_test, 255.0)

    fit(net, [x_train], [y_train], batch_size, epochs)
    evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    main()
