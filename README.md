# Megakernels!

## Installation

Clone this repo and run:

```bash
git submodule update --init --recursive
pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
```

## Low-Latency Llama Demo

To compile the megakernel, run:

```bash

# from the repo root
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.12 # adjust if yours is different
export GPU=H100 # options are {H100, A100, 4090}, else defaults to B200
cd demos/low-latency-llama
make

```

To start an interactive chat session, run:

```bash

# from the repo root
python megakernels/scripts/llama_repl.py

```

To benchmark the megakernel, run:

```bash

# from the repo root
python megakernels/scripts/generate.py mode=mk prompt="tell me a funny joke about cookies" ntok=100

```

## High-Throughput Llama Demo

To benchmark the megakernel, run:

```bash

# from the repo root
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.12 # adjust if yours is different
cd demos/high-throughput-llama
make

cd ../../
python megakernels/scripts/generate.py mode=mk .th prompt="tell me a funny joke about cookies" ntok=100

```

#### TEMP

```bash
# Diff test individual ops
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 stop_after_op=attn_norm 
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 start_after_op=attn_norm stop_after_op=qkv
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 start_after_op=qkv stop_after_op=attn 
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 start_after_op=attn stop_after_op=oproj
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 start_after_op=oproj stop_after_op=mlp_norm
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 start_after_op=mlp_norm stop_after_op=gate
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 start_after_op=gate stop_after_op=up
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 start_after_op=up

# Diff test the model
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 layer_limit=1
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 layer_limit=2
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 layer_limit=3
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 layer_limit=4
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 layer_limit=5
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 layer_limit=None
CUDA_VISIBLE_DEVICES=0 python megakernels/scripts/diff_test.py .th_h100 start_after_op=gate stop_after_op=up

# Generate
python megakernels/scripts/generate.py mode=mk .th_h100 prompt="tell me a funny joke about cookies " ntok=20

# Token/sec
CUDA_VISIBLE_DEVICES=5 python megakernels/scripts/generate.py .th_h100 mode=mk ntok=5 tokens=F prompt='("hi hi hi" * 5)'
```


## Tensor-parallel Llama Demo

To benchmark the megakernel, run:

```bash

# from the repo root
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.12 # adjust if yours is different
cd demos/cross-gpu-llama
make
```

Run commands:
```bash
# Run with python as:
torchrun --nproc_per_node=8 megakernels/scripts/generate.py mode=torch .tp_h100  prompt="tell me a funny joke about cookies " ntok=20

# Run with pyvm as:
torchrun --nproc_per_node=8 megakernels/scripts/generate.py mode=pyvm .tp_h100  prompt="tell me a funny joke about cookies " ntok=20

# Or:
python megakernels/scripts/tp_generate_pyvm.py

# Run with mk as:
python megakernels/scripts/tp_generate.py
```



