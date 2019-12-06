import argparse
import sys
import pyeddl._core.eddlT as eddlT


def main(args):
    dev = eddlT.DEV_GPU if args.gpu else eddlT.DEV_CPU
    A = eddlT.create([10, 10])
    eddlT.info(A)
    T = eddlT.randn([10, 10], dev)
    eddlT.print(T)
    eddlT.normalize_(T, 0, 1)
    eddlT.print(T)
    U = eddlT.randn([10, 3], dev)
    eddlT.print(U)
    V = eddlT.mult2D(T, U)
    eddlT.info(V)
    eddlT.print(V)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
