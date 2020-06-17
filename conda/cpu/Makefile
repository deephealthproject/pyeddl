SHELL := /bin/bash

all: build

build: eddl-conda

eddl-conda: Dockerfile.eddl
	docker build -t eddl-conda -f Dockerfile.eddl .

eddl-conda-test: Dockerfile.eddl-test eddl-conda
	${SHELL} make_repo.sh
	docker build -t eddl-conda-test -f Dockerfile.eddl-test .

test: eddl-conda-test
	docker run --rm eddl-conda-test bash test.sh

.PHONY: all build eddl-conda eddl-conda-test test
