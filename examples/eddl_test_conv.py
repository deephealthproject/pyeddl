from pyeddl._core import Tensor, ConvolDescriptor, Conv2D, Conv2D_grad
from pyeddl.api import DEV_CPU, DEV_GPU


class TestTensor(object):

    def __init__(self, shape):
        self.TC = Tensor(shape, DEV_CPU)
        self.TG = Tensor(shape, DEV_GPU)
        self.T = Tensor(shape, DEV_CPU)

    def to_gpu(self):
        Tensor.copy(self.TC, self.TG)

    def check(self, s):
        Tensor.copy(self.TG, self.T)
        res = "OK" if Tensor.equal(self.TC, self.T) else "Fail"
        print("%s: %s" % (s, res))


def check_c_vs_g(A, B, s):
    C = Tensor(B.getShape(), DEV_CPU)
    Tensor.copy(B, C)
    res = "OK" if Tensor.equal(A, C) else "Fail"
    print("%s: %s" % (s, res))


def main():
    A = TestTensor([1, 2, 5, 5])
    CDC = ConvolDescriptor([3, 3, 3], [2, 2], [1, 1])
    CDG = ConvolDescriptor([3, 3, 3], [2, 2], [1, 1])
    CDC.build(A.TC)
    CDG.build(A.TG)
    CDC.I.info()
    CDG.I.info()

    print("FORW")
    CDC.I.rand_signed_uniform(0.1)
    CDC.K.rand_signed_uniform(0.1)
    CDC.bias.rand_signed_uniform(0.1)
    Tensor.copy(CDC.I, CDG.I)
    Tensor.copy(CDC.K, CDG.K)
    Tensor.copy(CDC.bias, CDG.bias)
    Conv2D(CDG)
    Conv2D(CDC)
    check_c_vs_g(CDC.O, CDG.O, "conv2d")

    print("GRAD")
    CDC.D.rand_signed_uniform(0.1)
    CDC.gK.set(0.0)
    CDC.gbias.set(0.0)
    Tensor.copy(CDC.D, CDG.D)
    CDG.gK.set(0.0)
    CDG.gbias.set(0.0)
    Conv2D_grad(CDC)
    Conv2D_grad(CDG)
    check_c_vs_g(CDC.gK, CDG.gK, "conv2d_grad gK")
    check_c_vs_g(CDC.gbias, CDG.gbias, "conv2d_grad gbias")

    print("BACK")
    CDC.ID = Tensor(A.TC.getShape(), DEV_CPU)
    CDG.ID = Tensor(A.TG.getShape(), DEV_GPU)
    CDC.D.rand_signed_uniform(0.1)
    CDC.ID.set(0.0)
    Tensor.copy(CDC.D, CDG.D)
    CDG.ID.set(0.0)
    # --- FIXME Conv2D_back hangs indefinitely ---
    # Conv2D_back(CDC)
    # Conv2D_back(CDG)
    # check_c_vs_g(CDC.ID, CDG.ID, "conv2d_back")


if __name__ == "__main__":
    main()
