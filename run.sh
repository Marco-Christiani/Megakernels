#!/bin/bash
set -eo pipefail
source .envrc
# export VIRTUAL_ENV=/tmp/venv
export VIRTUAL_ENV="${VIRTUAL_ENV:-${MEGAKERNELS_ROOT}/.venv}"
cd $MEGAKERNELS_ROOT

if [ ! -d ${VIRTUAL_ENV} ]; then
    uv venv --python 3.12 "$VIRTUAL_ENV"
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi
source $VIRTUAL_ENV/bin/activate
uv pip install -e .
# cd linear_training_mk_demo/
# uv pip install -e .
# uv pip install pytest
# pytest

# cd demos/low-latency-llama
# make clean
# make
# cd $THUNDERKITTENS_ROOT
# python megakernels/scripts/llama_repl.py
