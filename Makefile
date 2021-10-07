SHELL := /bin/bash

eddl-base: docker/Dockerfile.eddl-base
	docker build -t dhealthdev/eddl-base -f $< .

eddl-cpu: docker/Dockerfile.eddl-cpu eddl-base
	docker build -t dhealthdev/eddl-cpu -f $< .

eddl-gpu: docker/Dockerfile.eddl-gpu eddl-base
	docker build -t dhealthdev/eddl-gpu -f $< .

eddl-cudnn: docker/Dockerfile.eddl-cudnn eddl-base
	docker build -t dhealthdev/eddl-cudnn -f $< .

pyeddl-cpu: docker/Dockerfile.pyeddl-cpu eddl-cpu
	docker build -t dhealthdev/pyeddl-cpu -f $< .

pyeddl-gpu: docker/Dockerfile.pyeddl-gpu eddl-gpu
	docker build -t dhealthdev/pyeddl-gpu -f $< .

pyeddl-cudnn: docker/Dockerfile.pyeddl-cudnn eddl-cudnn
	docker build -t dhealthdev/pyeddl-cudnn -f $< .

docs: docker/Dockerfile.docs pyeddl-cpu
	docker build -t dhealthdev/pyeddl-docs -f $< .

test-pyeddl-cpu: pyeddl-cpu
	docker run --rm dhealthdev/pyeddl-cpu pytest tests

test-pyeddl-gpu: pyeddl-gpu
	docker run --rm dhealthdev/pyeddl-gpu pytest tests

test-pyeddl-cudnn: pyeddl-cudnn
	docker run --rm dhealthdev/pyeddl-cudnn pytest tests

get-docs: docs
	rm -rf /tmp/html && docker run --rm dhealthdev/pyeddl-docs bash -c "tar -c -C /pyeddl/docs/source/_build html" | tar -x -C /tmp
