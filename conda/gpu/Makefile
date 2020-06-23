SHELL := /bin/bash
DEVICE := 1

eddl-conda: Dockerfile.eddl
	docker build -t eddl-conda -f Dockerfile.eddl .

eddl-conda-test: Dockerfile.eddl-test eddl-conda
	${SHELL} make_repo.sh
	docker build -t eddl-conda-test -f Dockerfile.eddl-test .

test-eddl: eddl-conda-test
	docker run --gpus "device=${DEVICE}" --rm eddl-conda-test bash test.sh

.PHONY: eddl-conda eddl-conda-test test-eddl
