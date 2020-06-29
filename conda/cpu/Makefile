SHELL := /bin/bash

eddl-conda: Dockerfile.eddl
	docker build -t eddl-conda -f Dockerfile.eddl .

eddl-conda-test: Dockerfile.eddl-test eddl-conda
	${SHELL} make_repo.sh
	docker build -t eddl-conda-test -f Dockerfile.eddl-test .

eddl-conda-test-cloud: Dockerfile.eddl-test-cloud
	docker build -t eddl-conda-test-cloud -f Dockerfile.eddl-test-cloud .

pyeddl-conda: Dockerfile.pyeddl eddl-conda-test
	docker build -t pyeddl-conda -f Dockerfile.pyeddl .

pyeddl-conda-test: Dockerfile.pyeddl-test pyeddl-conda
	${SHELL} add_pyeddl_to_repo.sh
	${SHELL} -c 'rm -rf tests && cp -rf ../../tests .'
	docker build -t pyeddl-conda-test -f Dockerfile.pyeddl-test .

pyeddl-conda-test-cloud: Dockerfile.pyeddl-test-cloud
	${SHELL} -c 'rm -rf tests && cp -rf ../../tests .'
	docker build -t pyeddl-conda-test-cloud -f Dockerfile.pyeddl-test-cloud .

test-eddl: eddl-conda-test
	docker run --rm eddl-conda-test bash test.sh

test-eddl-cloud: eddl-conda-test-cloud
	docker run --rm eddl-conda-test-cloud bash test.sh

test-pyeddl: pyeddl-conda-test
	docker run --rm pyeddl-conda-test bash test.sh

test-pyeddl-cloud: pyeddl-conda-test-cloud
	docker run --rm pyeddl-conda-test-cloud bash test.sh

.PHONY: eddl-conda eddl-conda-test pyeddl-conda pyeddl-conda-test test-eddl test-pyeddl test-eddl-cloud test-pyeddl-cloud
