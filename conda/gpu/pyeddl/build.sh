#!/bin/bash

export CPATH="${PREFIX}/include/eigen3:${CPATH}"
export EDDL_WITH_CUDA="true"
${PYTHON} setup.py install
