#!/bin/bash
# Usage: scripts/run_eval.sh /path/to/ckpt_dir
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-7500}
TOKENIZER=${TOKENIZER:-assets/zen_tokenizer}
DATA_ROOT=${DATA_ROOT:-/data2/datasets/eval_data}
LOG_DIR=${LOG_DIR:-logs/eval_$(date +%Y%m%d_%H%M%S)}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/ckpt_dir"
  exit 1
fi

CKPT=$1

COMMON_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --use-mcore-models
  --sequence-parallel
  --micro-batch-size 1
  --global-batch-size 1
  --seq-length 4096
  --max-new-tokens 64
  --evaluation-batch-size 1
  --tokenizer-name-or-path "${TOKENIZER}"
  --num-layers 24
  --hidden-size 896
  --ffn-hidden-size 4864
  --num-attention-heads 16
  --max-position-embeddings 4096
  --hidden-dropout 0.0
  --attention-dropout 0.0
  --position-embedding-type rope
  --use-flash-attn
  --no-masked-softmax-fusion
  --attention-softmax-in-fp32
  --swiglu
  --use-fused-swiglu
  --disable-bias-linear
  --normalization RMSNorm
  --use-fused-rmsnorm
  --untie-embeddings-and-output-weights
  --tokenizer-type PretrainedFromHF
  --bf16
  --use-kv-cache
  --use-fused-rotary-pos-emb
)

mkdir -p "${LOG_DIR}"

TASK="mmlu"
ASCEND_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node 1 --nnodes 1 --node_rank 0 \
  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT_BASE}" \
  evaluation.py "${COMMON_ARGS[@]}" \
  --add-eos-token $'\n' $'\n\n' \
  --prompt-type zen \
  --load "${CKPT}" \
  --task $TASK --task-data-path "${MMLU_DATA:-${DATA_ROOT}/$TASK/test}" --eval-language en \
  2>&1 | tee "${LOG_DIR}/$TASK.log"
