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
    dim1, dim2, dim3 = 1000, 1000, 100

    A = TestTensor([dim1, dim3])
    B = TestTensor([dim3, dim2])
    Bt = TestTensor([dim1, dim2])
    Bt2 = TestTensor([dim2, dim3])
    C = TestTensor([dim1, dim2])
    Ct = TestTensor([dim3, dim2])
    Ct2 = TestTensor([dim1, dim2])
    D = TestTensor([dim1, dim3])
    E = TestTensor([dim1, dim3])
    F = TestTensor([dim3])

    A.TC.rand_uniform(1.0)
    A.to_gpu()
    A.check("copy")

    A.TC.set(1.0)
    A.TG.set(1.0)
    A.check("set")

    A.TC.rand_signed_uniform(1)
    A.to_gpu()
    exp = 2.0
    A.TC.pow(exp)
    A.TG.pow(exp)
    A.check("pow")

    A.TC.rand_signed_uniform(1)
    A.to_gpu()
    fc = A.TC.sum()
    fg = A.TG.sum()
    res = "OK" if abs(fc - fg) <= 0.01 else "Fail"
    print("sum: %s" % res)

    A.TC.rand_uniform(1.0)
    B.TC.rand_uniform(1.0)
    A.to_gpu()
    B.to_gpu()
    Tensor.mult2D(A.TC, 0, B.TC, 0, C.TC, 0)
    Tensor.mult2D(A.TG, 0, B.TG, 0, C.TG, 0)
    C.check("mult2D")

    A.TC.rand_uniform(1.0)
    Bt.TC.rand_uniform(1.0)
    A.to_gpu()
    Bt.to_gpu()
    Tensor.mult2D(A.TC, 1, Bt.TC, 0, Ct.TC, 0)
    Tensor.mult2D(A.TG, 1, Bt.TG, 0, Ct.TG, 0)
    Ct.check("mult2D Trasp")

    A.TC.rand_uniform(1.0)
    Bt2.TC.rand_uniform(1.0)
    A.to_gpu()
    Bt2.to_gpu()
    Tensor.mult2D(A.TC, 0, Bt2.TC, 1, Ct2.TC, 0)
    Tensor.mult2D(A.TG, 0, Bt2.TG, 1, Ct2.TG, 0)
    Ct2.check("mult2D Trasp2")

    A.TC.rand_uniform(1.0)
    Bt2.TC.rand_uniform(1.0)
    A.to_gpu()
    Bt2.to_gpu()
    Tensor.mult2D(A.TC, 0, Bt2.TC, 1, Ct2.TC, 1)
    Tensor.mult2D(A.TG, 0, Bt2.TG, 1, Ct2.TG, 1)
    Ct2.check("mult2D Trasp2 inc")

    A.TC.rand_uniform(1.0)
    D.TC.rand_uniform(1.0)
    A.to_gpu()
    D.to_gpu()
    Tensor.add(1.0, A.TC, 1.0, D.TC, E.TC, 0)
    Tensor.add(1.0, A.TG, 1.0, D.TG, E.TG, 0)
    E.check("add")

    A.TC.rand_uniform(100.0)
    D.TC.rand_uniform(100.0)
    A.to_gpu()
    D.to_gpu()
    Tensor.inc(A.TC, D.TC)
    Tensor.inc(A.TG, D.TG)
    D.check("inc")

    A.TC.rand_signed_uniform(100000)
    A.to_gpu()
    Tensor.Softmax(A.TC, D.TC)
    Tensor.Softmax(A.TG, D.TG)
    D.check("Softmax")

    A.TC.rand_uniform(1)
    D.TC.rand_binary(0.1)
    A.to_gpu()
    D.to_gpu()
    Tensor.cent(A.TC, D.TC, E.TC)
    Tensor.cent(A.TG, D.TG, E.TG)
    E.check("cross entropy")

    A.TC.rand_uniform(1.0)
    F.TC.rand_uniform(1.0)
    A.to_gpu()
    F.to_gpu()
    Tensor.sum2D_rowwise(A.TC, F.TC, D.TC)
    Tensor.sum2D_rowwise(A.TG, F.TG, D.TG)
    D.check("sum2D_rowwise")

    A.TC.rand_uniform(1.0)
    F.TC.rand_uniform(1.0)
    A.to_gpu()
    F.to_gpu()
    Tensor.reduce_sum2D(A.TC, F.TC, 0, 0)
    Tensor.reduce_sum2D(A.TG, F.TG, 0, 0)
    F.check("reduce_sum2D")

    A.TC.rand_uniform(1.0)
    F.TC.rand_uniform(1.0)
    A.to_gpu()
    F.to_gpu()
    Tensor.reduce_sum2D(A.TC, F.TC, 0, 1)
    Tensor.reduce_sum2D(A.TG, F.TG, 0, 1)
    F.check("reduce_sum2D inc")

    A.TC.rand_signed_uniform(1.0)
    A.to_gpu()
    Tensor.ReLu(A.TC, D.TC)
    Tensor.ReLu(A.TG, D.TG)
    D.check("ReLu")

    A.TC.rand_signed_uniform(1.0)
    D.TC.rand_signed_uniform(1.0)
    A.to_gpu()
    D.to_gpu()
    Tensor.D_ReLu(D.TC, A.TC, E.TC)
    Tensor.D_ReLu(D.TG, A.TG, E.TG)
    E.check("D_ReLu")


if __name__ == "__main__":
    main()
