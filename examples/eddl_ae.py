from pyeddl.api import (
    Input, Activation, Dense, Model, sgd, CS_CPU, build, T_load, div, fit
)
from pyeddl.utils import download_mnist


def main():
    download_mnist()

    epochs = 10
    batch_size = 1000

    in_ = Input([784])
    layer = in_
    layer = Activation(Dense(layer, 256), "relu")
    layer = Activation(Dense(layer, 128), "relu")
    layer = Activation(Dense(layer, 64), "relu")
    layer = Activation(Dense(layer, 128), "relu")
    layer = Activation(Dense(layer, 256), "relu")
    out = Dense(layer, 784)
    net = Model([in_], [out])
    print(net.summary())

    build(
        net,
        sgd(0.01, 0.9),
        ["mean_squared_error"],
        ["mean_squared_error"],
        CS_CPU(4)
    )

    x_train = T_load("trX.bin")
    div(x_train, 255.0)
    fit(net, [x_train], [x_train], batch_size, epochs)


if __name__ == "__main__":
    main()
