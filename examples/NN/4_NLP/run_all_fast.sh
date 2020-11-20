#!/usr/bin/env bash

# Assumes EDDL compiled for GPU.

set -euo pipefail
[ -n "${DEBUG:-}" ] && set -x
this="${BASH_SOURCE-$0}"
this_dir=$(cd -P -- "$(dirname -- "${this}")" && pwd -P)

names=(
    nlp_sentiment_rnn
    nlp_machine_translation
    nlp_text_generation
)

for n in "${names[@]}"; do
    echo -en "\n*** ${n} ***\n"
    python3 "${this_dir}"/${n}.py --gpu --epochs 1 --small
done
