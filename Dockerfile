FROM crs4/cmake:3.14

RUN apt-get -y update && apt-get -y install --no-install-recommends \
      python3-dev \
      python3-pip \
      wget

RUN python3 -m pip install --upgrade --no-cache-dir \
      setuptools pip numpy

# Run git submodule update [--init] --recursive first
COPY . /pyeddl

RUN cd /pyeddl/third_party/eddl && \
    patch -p1 < ../../compserv.diff && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(grep -c ^processor /proc/cpuinfo) && \
    make install

WORKDIR /pyeddl

RUN bash build.sh
