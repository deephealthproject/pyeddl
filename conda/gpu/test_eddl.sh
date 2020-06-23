#!/usr/bin/env bash

set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate test
cat <<EOF >example.cpp
#include <eddl/apis/eddl.h>
#include <eddl/apis/eddlT.h>

int main() {
    eddl::download_mnist();
    int epochs = 1;
    int batch_size = 100;
    eddl::layer in = eddl::Input({784});
    eddl::layer l = in;
    l = eddl::Activation(eddl::Dense(l, 128), "relu");
    l = eddl::Activation(eddl::Dense(l, 64), "relu");
    l = eddl::Activation(eddl::Dense(l, 128), "relu");
    eddl::layer out = eddl::Dense(l, 784);
    eddl::model net = eddl::Model({in}, {out});
    eddl::build(net, eddl::sgd(0.0001, 0.89),
		{"mean_squared_error"}, {"mean_squared_error"},
		eddl::CS_GPU({1}, "low_mem"));
    eddl::summary(net);
    eddl::plot(net, "model.pdf");
    tensor x_train = eddlT::load("trX.bin");
    eddlT::div_(x_train, 255.0);
    eddl::fit(net, {x_train}, {x_train}, batch_size, epochs);
}
EOF

# conda install -y gxx_linux-64==7.3.0
# x86_64-conda_cos6-linux-gnu-g++ -I/opt/conda/envs/test/include -I/opt/conda/envs/test/include/eigen3 -L /opt/conda/envs/test/lib example.cpp -o example -std=c++11 -leddl -pthread

apt-get -y install --no-install-recommends g++
g++ -I/opt/conda/envs/test/include -I/opt/conda/envs/test/include/eigen3 -L /opt/conda/envs/test/lib example.cpp -o example -std=c++11 -leddl -pthread
export LD_LIBRARY_PATH="/opt/conda/envs/test/lib:${LD_LIBRARY_PATH}"
./example
