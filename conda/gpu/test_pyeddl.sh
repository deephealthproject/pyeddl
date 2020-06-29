#!/usr/bin/env bash

set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
for v in 3.6 3.7 3.8; do
    conda activate test${v}
    conda install -y pytest
    pytest /tests
    conda deactivate
done
