import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT


def main():
    eddl.download_mnist()

    epochs = 1
    batch_size = 1000

    in_ = eddl.Input([784])

    layer = in_
    layer = eddl.Activation(eddl.Dense(layer, 256), "relu")
    layer = eddl.Activation(eddl.Dense(layer, 128), "relu")
    layer = eddl.Activation(eddl.Dense(layer, 64), "relu")
    layer = eddl.Activation(eddl.Dense(layer, 128), "relu")
    layer = eddl.Activation(eddl.Dense(layer, 256), "relu")
    out = eddl.Dense(layer, 784)

    net = eddl.Model([in_], [out])
    eddl.build(
        net,
        eddl.sgd(0.001, 0.9),
        ["mean_squared_error"],
        ["mean_squared_error"],
        eddl.CS_CPU(4)
    )

    print(eddl.summary(net))
    eddl.plot(net, "model.pdf")

    x_train = eddlT.load("trX.bin")
    eddlT.div_(x_train, 255.0)
    eddl.fit(net, [x_train], [x_train], batch_size, epochs)


if __name__ == "__main__":
    main()
