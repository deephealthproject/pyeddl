#!/usr/bin/env bash

set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate test
conda install pytest
pytest /tests
