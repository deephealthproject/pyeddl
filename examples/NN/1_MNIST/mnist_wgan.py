# Copyright (c) 2019-2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""\
Wasserstein GAN for MNIST.
"""

import argparse
import sys

import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT


def vreal_loss(in_):
    # maximize for real images (mimize -1 x Value); in_ = vector of layers
    return eddl.ReduceMean(eddl.Mult(in_[0], -1))


def vfake_loss(in_):
    # minimizes for fake images; in_ = single layer
    return eddl.ReduceMean(in_)


def main(args):
    eddl.download_mnist()

    # Define Generator
    gin = eddl.GaussGenerator(0.0, 1, [100])
    layer = gin
    layer = eddl.LReLu(eddl.Dense(layer, 256))
    layer = eddl.LReLu(eddl.Dense(layer, 512))
    layer = eddl.LReLu(eddl.Dense(layer, 1024))
    gout = eddl.Tanh(eddl.Dense(layer, 784))

    gen = eddl.Model([gin], [])
    gopt = eddl.rmsprop(0.001)
    eddl.build(gen, gopt)  # CS_CPU by default
    if args.gpu:
        eddl.toGPU(gen)  # GPU [1] by default

    # Define Discriminator
    din = eddl.Input([784])
    layer = din
    layer = eddl.LReLu(eddl.Dense(layer, 1024))
    layer = eddl.LReLu(eddl.Dense(layer, 512))
    layer = eddl.LReLu(eddl.Dense(layer, 256))
    dout = eddl.Dense(layer, 1)

    disc = eddl.Model([din], [])
    dopt = eddl.rmsprop(0.001)
    eddl.build(disc, dopt)  # CS_CPU by default
    if args.gpu:
        eddl.toGPU(disc)  # GPU [1] by default

    eddl.summary(gen)
    eddl.summary(disc)

    # Load dataset
    x_train = eddlT.load("trX.bin")
    # Preprocessing [-1, 1]
    eddlT.div_(x_train, 128.0)
    eddlT.sub_(x_train, 1.0)

    # Training
    batch = eddlT.create([args.batch_size, 784])
    # Wasserstein GAN params
    critic = 5
    clip = 0.01

    rl = eddl.newloss(vreal_loss, [dout], "real_loss")
    fl = eddl.newloss(vfake_loss, dout, "fake_loss")
    for i in range(args.epochs):
        print("Epoch %d/%d (%d batches)" %
              (i + 1, args.epochs, args.num_batches))
        for j in range(args.num_batches):
            for k in range(critic):
                # get a batch from real images
                eddl.next_batch([x_train], [batch])
                # Train Discriminator
                eddl.zeroGrads(disc)
                # Real
                eddl.forward(disc, [batch])
                dr = eddl.compute_loss(rl)
                eddl.backward(rl)
                # Fake
                eddl.forward(disc, eddl.detach(eddl.forward(
                    gen, args.batch_size
                )))
                df = eddl.compute_loss(fl)
                eddl.backward(fl)
                eddl.update(disc)
                eddl.clamp(disc, -clip, clip)
            # Train Gen
            eddl.zeroGrads(gen)
            eddl.forward(disc, eddl.forward(gen, args.batch_size))
            gr = eddl.compute_loss(rl)
            eddl.backward(rl)
            eddl.update(gen)
            print("Batch %d - Total Loss=%1.3f - Dr=%1.3f Df=%1.3f Gr=%1.3f" %
                  (j+1, dr + df + gr, dr, df, gr))

    # Generate some num_samples
    eddl.forward(gen, args.batch_size)
    output = eddl.getTensor(gout)
    img = eddlT.select(output, 0)
    eddlT.reshape_(img, [1, 1, 28, 28])
    eddlT.save(img, "./img.png", "png")
    img1 = eddlT.select(output, 5)
    eddlT.reshape_(img1, [1, 1, 28, 28])
    eddlT.save(img1, "./img1.png", "png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--num-batches", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
