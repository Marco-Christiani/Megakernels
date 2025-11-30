#!/bin/bash
set -eo pipefail

print_help() {
    cat <<'EOF'
Usage: run_linear_training.sh [options] [-- <extra make args>]
Options:
  --debug       Rebuild with debug instrumentation, enable verbose CUDA/Python logging (implies --timings).
  --timings     Enable timing capture/reporting (sets LINEAR_DEMO_TIMINGS=1 and ENABLE_TIMINGS=1).
  --nobuild     Skip the make step; assumes the extension is already built.
  --mode=MODE   Override the Makefile MODE (opt, debug, quick).
  --gpu=GPU     Override the GPU target (H100, A100, 4090, etc.).
  -h, --help    Show this message.

All other arguments after '--' are forwarded directly to make.
EOF
}

# Defaults
MODE="${MODE:-opt}"
ENABLE_TIMINGS="${ENABLE_TIMINGS:-0}"
ENABLE_DEBUG="${ENABLE_DEBUG:-0}"
LINEAR_DEMO_DEBUG="${LINEAR_DEMO_DEBUG:-0}"
LINEAR_DEMO_TIMINGS="${LINEAR_DEMO_TIMINGS:-0}"
GPU="${GPU:-H100}"

BUILD=1
MAKE_ARGS=()
MODE_OVERRIDE=0

while (($#)); do
    case "$1" in
        --debug)
            LINEAR_DEMO_DEBUG=1
            LINEAR_DEMO_TIMINGS=1
            ENABLE_TIMINGS=1
            ENABLE_DEBUG=1
            if [ "${MODE_OVERRIDE}" -eq 1 ] && [ "${MODE}" != "debug" ]; then
                echo "[linear-demo] WARNING: overriding MODE=${MODE} with MODE=debug due to --debug"
            fi
            MODE="debug"
            shift
            ;;
        --timings)
            LINEAR_DEMO_TIMINGS=1
            ENABLE_TIMINGS=1
            shift
            ;;
        --nobuild)
            BUILD=0
            shift
            ;;
        --mode=*)
            MODE="${1#*=}"
            MODE_OVERRIDE=1
            shift
            ;;
        --gpu=*)
            GPU="${1#*=}"
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        --)
            shift
            MAKE_ARGS+=("$@")
            break
            ;;
        *)
            MAKE_ARGS+=("$1")
            shift
            ;;
    esac
done

# Export for downstream steps
export MODE
export ENABLE_TIMINGS
export ENABLE_DEBUG
export LINEAR_DEMO_DEBUG
export LINEAR_DEMO_TIMINGS
export GPU

# sequence is envrc -> uv venv -> pip install -> make -> run

if [ -f .envrc ]; then
    # shellcheck disable=SC1091
    source .envrc
fi

export MEGAKERNELS_ROOT="${MEGAKERNELS_ROOT:-$(pwd)}"
export THUNDERKITTENS_ROOT="${THUNDERKITTENS_ROOT:-${MEGAKERNELS_ROOT}/ThunderKittens}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-${MEGAKERNELS_ROOT}/.venv}"
# export VIRTUAL_ENV="${VIRTUAL_ENV:-/tmp/venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

cd "${MEGAKERNELS_ROOT}"

if [ ! -d ${VIRTUAL_ENV} ]; then
    uv venv --python "${PYTHON_VERSION}" "$VIRTUAL_ENV"
    source "${VIRTUAL_ENV}/bin/activate"
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    uv pip install pybind11
    uv pip install -e .
else
    source "${VIRTUAL_ENV}/bin/activate"
fi


cd demos/linear-training
set -x
# if [ -z "$@" ]; then
#     make GPU="${GPU}" PYTHON_VERSION="${PYTHON_VERSION}" PYTHON_BIN="$(which python)" PYTHON_CONFIG="python${PYTHON_VERSION}-config"
# else
# fi

if [ "${BUILD}" -eq 1 ]; then
    time make ENABLE_TIMINGS="${ENABLE_TIMINGS}" ENABLE_DEBUG="${ENABLE_DEBUG}" MODE="${MODE}" GPU="${GPU}" PYTHON_VERSION="${PYTHON_VERSION}" PYTHON_BIN="$(which python)" PYTHON_CONFIG="python${PYTHON_VERSION}-config" "${MAKE_ARGS[@]}"
fi

python run_linear_training.py
