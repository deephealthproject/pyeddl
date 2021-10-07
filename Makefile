SHELL := /bin/bash

eddl-base: docker/Dockerfile.eddl-base
	docker build -t dhealthdev/eddl-base -f $< .

eddl-cpu: docker/Dockerfile.eddl-cpu eddl-base
	docker build -t dhealthdev/eddl-cpu -f $< .

eddl-gpu: docker/Dockerfile.eddl-gpu eddl-base
	docker build -t dhealthdev/eddl-gpu -f $< .

eddl-cudnn: docker/Dockerfile.eddl-cudnn eddl-base
	docker build -t dhealthdev/eddl-cudnn -f $< .
