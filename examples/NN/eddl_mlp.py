import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT


def main():
    eddl.download_mnist()

    epochs = 1
    batch_size = 1000
    num_classes = 10

    in_ = eddl.Input([784])

    layer = in_
    layer = eddl.BatchNormalization(
        eddl.Activation(eddl.L2(eddl.Dense(layer, 1024), 0.0001), "relu")
    )
    layer = eddl.BatchNormalization(
        eddl.Activation(eddl.L2(eddl.Dense(layer, 1024), 0.0001), "relu")
    )
    layer = eddl.BatchNormalization(
        eddl.Activation(eddl.L2(eddl.Dense(layer, 1024), 0.0001), "relu")
    )
    out = eddl.Activation(eddl.Dense(layer, num_classes), "softmax")
    net = eddl.Model([in_], [out])

    eddl.plot(net, "model.pdf")

    eddl.build(
        net,
        eddl.sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_CPU(4)
    )

    print(eddl.summary(net))

    x_train = eddlT.load("trX.bin")
    y_train = eddlT.load("trY.bin")
    x_test = eddlT.load("tsX.bin")
    y_test = eddlT.load("tsY.bin")

    eddlT.div_(x_train, 255.0)
    eddlT.div_(x_test, 255.0)

    for i in range(epochs):
        eddl.fit(net, [x_train], [y_train], batch_size, 1)
        eddl.evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    main()
