#!/bin/bash

export CPATH="${PREFIX}/include/eigen3:${CPATH}"
${PYTHON} setup.py install
