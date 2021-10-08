SHELL := /bin/bash

eddl-base: docker/Dockerfile.eddl-base
	docker build -t dhealth/dev-eddl-base -f $< .

eddl-cpu: docker/Dockerfile.eddl-cpu eddl-base
	docker build -t dhealth/dev-eddl-cpu -f $< .

eddl-gpu: docker/Dockerfile.eddl-gpu eddl-base
	docker build -t dhealth/dev-eddl-gpu -f $< .

eddl-cudnn: docker/Dockerfile.eddl-cudnn eddl-base
	docker build -t dhealth/dev-eddl-cudnn -f $< .

pyeddl-cpu: docker/Dockerfile.pyeddl-cpu eddl-cpu
	docker build -t dhealth/dev-pyeddl-cpu -f $< .

pyeddl-gpu: docker/Dockerfile.pyeddl-gpu eddl-gpu
	docker build -t dhealth/dev-pyeddl-gpu -f $< .

pyeddl-cudnn: docker/Dockerfile.pyeddl-cudnn eddl-cudnn
	docker build -t dhealth/dev-pyeddl-cudnn -f $< .

docs: docker/Dockerfile.docs pyeddl-cpu
	docker build -t dhealth/dev-pyeddl-docs -f $< .

test-pyeddl-cpu: pyeddl-cpu
	docker run --rm dhealth/dev-pyeddl-cpu pytest tests

test-pyeddl-gpu: pyeddl-gpu
	docker run --rm dhealth/dev-pyeddl-gpu pytest tests

test-pyeddl-cudnn: pyeddl-cudnn
	docker run --rm dhealth/dev-pyeddl-cudnn pytest tests

get-docs: docs
	rm -rf /tmp/html && docker run --rm dhealth/dev-pyeddl-docs bash -c "tar -c -C /pyeddl/docs/source/_build html" | tar -x -C /tmp
