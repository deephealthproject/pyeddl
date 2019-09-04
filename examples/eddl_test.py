from pyeddl._core import Tensor
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
        res = "OK" if Tensor.equal(self.T, self.TC) else "Fail"
        print("%s: %s" % (s, res))


def main():
    A = TestTensor([10, 10])
    B = TestTensor([10, 100])
    C = TestTensor([10, 100])
    D = TestTensor([10, 10])
    E = TestTensor([10, 10])

    A.TC.rand_uniform(1.0)
    A.to_gpu()
    A.check("copy")

    A.TC.set(1.0)
    A.TG.set(1.0)
    A.check("set")

    A.TC.rand_uniform(1.0)
    B.TC.rand_uniform(1.0)
    A.to_gpu()
    B.to_gpu()
    Tensor.mult2D(A.TC, 0, B.TC, 0, C.TC, 0)
    Tensor.mult2D(A.TG, 0, B.TG, 0, C.TG, 0)
    C.check("mult2D")

    A.TC.rand_uniform(1.0)
    D.TC.rand_uniform(1.0)
    A.to_gpu()
    D.to_gpu()
    Tensor.sum(1.0, A.TC, 1.0, D.TC, E.TC, 0)
    Tensor.sum(1.0, A.TG, 1.0, D.TG, E.TG, 0)
    E.check("sum")


if __name__ == "__main__":
    main()
