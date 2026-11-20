#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CKPT_DIR="outputs/zen_500m_20k"
CKPT_LOAD_ITER=10000
TOKENIZER_MODEL="assets/zen_tokenizer"
TP=1
PP=1
CP=1
TOKENS_PER_STEP=524288
TRAIN_ITERS=20000
BATCH_LENGTH=4096
BATCH_SIZE_PER_DEVICE=4
CKPT_SAVE_DIR="${CKPT_DIR}_resume_$(expr $CKPT_LOAD_ITER/1000)k"


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


    # --context-parallel-size ${CP} \
    # --context-parallel-algo ulysses_cp_algo \
GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --use-mcore-models \
    --make-vocab-size-divisible-by 128 \
    --num-layers 24 \
    --hidden-size 896 \
    --hidden-dropout 0.0 \
    --num-attention-heads 16 \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --attention-dropout 0.0 \
    --ffn-hidden-size 4864 \
    --swiglu \
    --use-fused-swiglu \
    --position-embedding-type rope \
    --use-fused-rotary-pos-emb \
    --disable-bias-linear \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --untie-embeddings-and-output-weights \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length ${BATCH_LENGTH} \
    --max-position-embeddings ${BATCH_LENGTH} \
    --micro-batch-size ${BATCH_SIZE_PER_DEVICE} \
    --global-batch-size $(expr $TOKENS_PER_STEP / $BATCH_LENGTH) \
    --lr 3e-4 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --init-method-std 0.01 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 65536 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --bf16
"

DATA_ARGS="
    --train-data-path 1.0 $DATA_PATH \
    --valid-data-path 1.0 $VAL_DATA_PATH \
    --test-data-path 1.0 $VAL_DATA_PATH
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
"

mkdir -p "$CKPT_SAVE_DIR"

CKPT_SUBDIR=$(printf "iter_%07d" "$CKPT_LOAD_ITER")
TMP_LOAD_DIR=$(mktemp -d /tmp/zen_500m_$(expr $CKPT_LOAD_ITER/1000)k_resume_load.XXXXXX)
cleanup() {
    rm -rf "$TMP_LOAD_DIR"
}
trap cleanup EXIT

ln -s "$(readlink -f ${CKPT_DIR}/${CKPT_SUBDIR})" "${TMP_LOAD_DIR}/${CKPT_SUBDIR}"
echo "$CKPT_LOAD_ITER" > "${TMP_LOAD_DIR}/latest_checkpointed_iteration.txt"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
conda run -n m2 torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --jit-compile \
    --load $TMP_LOAD_DIR \
    --save $CKPT_SAVE_DIR \
    --ckpt-step $CKPT_LOAD_ITER \
    | tee logs/zen_500m_$(expr $CKPT_LOAD_ITER/1000)k_resume_$(expr $TRAIN_ITERS/1000)k_$(date +%Y%m%d_%H%M%S).log
