#!/usr/bin/env bash

# Assumes EDDL compiled for GPU.

set -euo pipefail
[ -n "${DEBUG:-}" ] && set -x
this="${BASH_SOURCE-$0}"
this_dir=$(cd -P -- "$(dirname -- "${this}")" && pwd -P)

names=(
    nlp_sentiment_rnn
    # nlp_text_generation
)

for n in "${names[@]}"; do
    echo -en "\n*** ${n} ***\n"
    python3 "${this_dir}"/${n}.py --gpu --epochs 1 --small
done

echo -en "\n*** nlp_video_to_labels ***\n"
python3 "${this_dir}"/nlp_video_to_labels.py --gpu --small

echo -en "\n*** nlp_machine_translation ***\n"
python3 "${this_dir}"/nlp_machine_translation.py --gpu --small

echo -en "\n*** nlp_sentiment_gru ***\n"
python3 "${this_dir}"/nlp_sentiment_gru.py --gpu --small
