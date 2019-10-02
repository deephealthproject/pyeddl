"""\
UNET example.
"""

import argparse
import sys

from pyeddl.api import (
    Input, Activation, Conv, MaxPool, Dropout, UpSampling
)


def main(args):

    dimX, dimY = 320, 320
    filters = 32
    ks = [3, 3]

    in_ = Input([3, dimX, dimY])
    l0 = in_
    l1 = MaxPool(Activation(Conv(l0, filters, ks), "relu"), [2, 2])
    l2 = MaxPool(Activation(Conv(l1, filters * 2, ks), "relu"), [2, 2])
    l3 = MaxPool(Activation(Conv(l2, filters * 4, ks), "relu"), [2, 2])
    l4 = MaxPool(Activation(Conv(l3, filters * 8, ks), "relu"), [2, 2])
    l5 = MaxPool(Activation(Conv(l4, filters * 16, ks), "relu"), [2, 2])
    l6 = Dropout(l5, 0.5)
    l7 = UpSampling(l6, [2, 2])

    # l7.output is None (In C++ l7->output is NULL), since LUpSampling is not
    # yet implemented (current implementation is only a stub). Due to this, the
    # LConcat constructor cannot work, since it tries to access the
    # ->output->ndim of each layer.

    # l8 = Concat([l4, l7])

    # l9 = Activation(Conv(l8, filters * 8, ks), "relu")
    # l10 = UpSampling(l9, [2, 2])
    # l11 = Concat([l3, l10])
    # l12 = Activation(Conv(l11, filters * 4, ks), "relu")
    # l13 = UpSampling(l12, [2, 2])
    # l14 = Concat([l2, l13])
    # l15 = Activation(Conv(l14, filters * 2, ks), "relu")
    # l16 = UpSampling(l15, [2, 2])
    # l17 = Concat([l1, l16])
    # l18 = Activation(Conv(l17, filters, ks), "relu")
    # l19 = Activation(Conv(l18, 1, [1, 1]), "softmax")  # should be "sigmoid"

    # out = Activation(Dense(layer, num_classes), "softmax")

    # net = Model([in_], [out])
    # print(net.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
