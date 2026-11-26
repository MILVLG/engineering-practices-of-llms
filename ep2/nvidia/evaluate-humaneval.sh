#!/usr/bin/env bash
set -euo pipefail
lesson="$(cd "$(dirname "$0")" && pwd)"
cd "${EP2_WORKDIR:?Set EP2_WORKDIR to your experiment directory}"
out="$1"
run="$2"
gpu="$3"
step="$4"
directory="$out/humaneval-$step"
mkdir "$directory"
checkpoint="$(realpath "$run")/checkpoints/checkpoint-$step"
export HF_HOME="$PWD/.cache/huggingface" HF_DATASETS_CACHE="$PWD/.cache/huggingface/datasets" HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="$PWD/.cache" MODELSCOPE_CACHE="$PWD/.cache/modelscope" TMPDIR="$PWD/.cache/tmp"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1
trap 'status=$?; printf "%s\n" "$status" >"$directory/exit-code"' EXIT
used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
test "$used" -lt 1000
.venv/bin/python "$lesson/evalscope_humaneval.py" eval \
    --model "$checkpoint" --model-id "pocket-sft-$step" --eval-type llm_ckpt \
    --model-args '{"precision":"bfloat16","attn_implementation":"sdpa"}' \
    --datasets humaneval --dataset-hub huggingface --dataset-dir "$PWD/.cache/evalscope-datasets" \
    --generation-config '{"do_sample":false,"max_tokens":1024}' --eval-batch-size 32 \
    --work-dir "$directory" --no-timestamp --seed 42 >"$directory/eval.log" 2>&1
