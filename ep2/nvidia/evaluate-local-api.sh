#!/usr/bin/env bash
set -euo pipefail
lesson="$(cd "$(dirname "$0")" && pwd)"
cd "${EP2_WORKDIR:?Set EP2_WORKDIR to your experiment directory}"
out="$1"
run="$2"
gpu="$3"
step="$4"
port="$5"
directory="$out/accelerated-$step"
mkdir "$directory"
checkpoint="$(realpath "$run")/checkpoints/checkpoint-$step"
model_id="pocket-sft-$step"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME="$PWD/.cache/huggingface" HF_DATASETS_CACHE="$PWD/.cache/huggingface/datasets" XDG_CACHE_HOME="$PWD/.cache" HF_HUB_DISABLE_XET=1
export MODELSCOPE_CACHE="$PWD/.cache/modelscope" TMPDIR="$PWD/.cache/tmp" NLTK_DATA="$PWD/.cache/nltk-data"
export TRITON_CACHE_DIR="$PWD/.cache/vllm-triton" VLLM_CACHE_ROOT="$PWD/.cache/vllm" VLLM_NO_USAGE_STATS=1
server=
trap 'code=$?; if test -n "$server"; then kill -TERM "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true; fi; printf "%s\n" "$code" >"$directory/exit-code"' EXIT
used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
test "$used" -lt 1000
.cache/venvs/pocket-vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port "$port" --model "$checkpoint" --served-model-name "$model_id" \
    --dtype bfloat16 --max-model-len 4096 --max-num-seqs 96 --gpu-memory-utilization .55 \
    --disable-log-requests >"$directory/server.log" 2>&1 &
server=$!
for i in $(seq 1 100); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then break; fi
    kill -0 "$server"
    sleep 2
done
curl -fsS "http://127.0.0.1:$port/health" >/dev/null
.venv/bin/python "$lesson/rescore-ifeval.py" eval --model "$model_id" --api-url "http://127.0.0.1:$port/v1/chat/completions" \
    --api-key local --eval-type openai_api --datasets gsm8k ifeval \
    --dataset-hub huggingface --dataset-dir "$PWD/.cache/evalscope-datasets" \
    --generation-config '{"do_sample":false,"temperature":0,"max_tokens":2048}' \
    --eval-batch-size 96 --work-dir "$directory" --no-timestamp --seed 42 >"$directory/eval.log" 2>&1
