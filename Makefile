SHELL := /bin/bash

all: pyeddl-cpu pyeddl-gpu pyeddl-cudnn

# images
eddl-base: docker/Dockerfile.eddl-base
	docker build -t dhealth/dev-eddl-base -f $< .

eddl-cpu: docker/Dockerfile.eddl-cpu eddl-base
	docker build -t dhealth/dev-eddl-cpu -f $< .

eddl-gpu: docker/Dockerfile.eddl-gpu eddl-base
	docker build -t dhealth/dev-eddl-gpu -f $< .

eddl-cudnn: docker/Dockerfile.eddl-cudnn eddl-base
	docker build -t dhealth/dev-eddl-cudnn -f $< .

pyeddl-base-cpu: docker/Dockerfile.pyeddl-base eddl-cpu
	docker build --build-arg target=cpu -t dhealth/dev-pyeddl-base-cpu -f $< .

pyeddl-base-gpu: docker/Dockerfile.pyeddl-base eddl-gpu
	docker build --build-arg target=gpu -t dhealth/dev-pyeddl-base-gpu -f $< .

pyeddl-base-cudnn: docker/Dockerfile.pyeddl-base eddl-cudnn
	docker build --build-arg target=cudnn -t dhealth/dev-pyeddl-base-cudnn -f $< .

pyeddl-cpu: docker/Dockerfile.pyeddl-cpu pyeddl-base-cpu
	docker build -t dhealth/dev-pyeddl-cpu -f $< .

pyeddl-gpu: docker/Dockerfile.pyeddl-gpu pyeddl-base-gpu
	docker build -t dhealth/dev-pyeddl-gpu -f $< .

pyeddl-cudnn: docker/Dockerfile.pyeddl-cudnn pyeddl-base-cudnn
	docker build -t dhealth/dev-pyeddl-cudnn -f $< .

docs: docker/Dockerfile.docs pyeddl-cpu
	docker build -t dhealth/dev-pyeddl-docs -f $< .

#commands
test-pyeddl-cpu: pyeddl-cpu
	docker run --rm dhealth/dev-pyeddl-cpu pytest tests

test-pyeddl-gpu: pyeddl-gpu
	docker run --rm dhealth/dev-pyeddl-gpu pytest tests

test-pyeddl-cudnn: pyeddl-cudnn
	docker run --rm dhealth/dev-pyeddl-cudnn pytest tests

get-docs: docs
	rm -rf /tmp/html && docker run --rm dhealth/dev-pyeddl-docs bash -c "tar -c -C /pyeddl/docs/source/_build html" | tar -x -C /tmp

.PHONY: all eddl-base eddl-cudnn get-docs pyeddl-base-cudnn pyeddl-cpu pyeddl-gpu test-pyeddl-cudnn docs eddl-cpu eddl-gpu pyeddl-base-cpu pyeddl-base-gpu pyeddl-cudnn test-pyeddl-cpu test-pyeddl-gpu
