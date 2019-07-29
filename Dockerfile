FROM eddl

RUN apt-get -y update && apt-get -y install --no-install-recommends \
      python3-dev \
      python3-pip

RUN python3 -m pip install --upgrade --no-cache-dir \
      setuptools pip numpy

# Run git submodule update [--init] --recursive first
COPY . /pyeddl

WORKDIR /pyeddl

RUN bash build_bindings.sh
