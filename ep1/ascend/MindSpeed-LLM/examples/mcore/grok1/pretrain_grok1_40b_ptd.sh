#!/bin/bash
# To check the performance of a Dropless MoE model, we should run the model for at least 500 iterations or resume from trained checkpoints.
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=4
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


TOKENIZER_MODEL="your tokenizer path"
DATA_PATH="your data path"
SAVE_PATH="your model save ckpt path"
LOAD_PATH="your model ckpt path"

TRANS_TYPE='local'


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
#grok1总层数是64
MODEL_ARGS="
    --use-mcore-models \
    --disable-bias-linear \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --num-layers 12 \
    --hidden-size 6144 \
    --ffn-hidden-size 32768 \
    --num-attention-heads 48 \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --position-embedding-type rope \
    --use-fused-swiglu \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --group-query-attention \
    --num-query-groups 8 \
    --use-flash-attn
"

MOE_ARGS="
    --num-experts 8 \
    --expert-model-parallel-size 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 1e-2 \
    --embedding-multiplier-scale 78.38367176906169 \
    --output-multiplier-scale 0.5773502691896257 \
    --input-jitter
"

DATA_ARGS="
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --data-path $DATA_PATH \
    --split 99990,8,2
"

TRAINING_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --lr 1e-5 \
    --train-iters 2000 \
    --lr-decay-iters 1280 \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 0.1 \
    --lr-warmup-iters 2 \
    --clip-grad 1.0 \
    --bf16 \
    --vocab-size 131072 \
    --use-distributed-optimizer \
    --no-rope-fusion \
    --no-bias-dropout-fusion \
    --no-bias-swiglu-fusion \
    --no-gradient-accumulation-fusion \
    --transformer-impl $TRANS_TYPE \
    --post-norm \
    --use-mc2 \
    --moe-permutation-async-comm \
    --no-shared-storage
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 2 \
    --sequence-parallel
"

LOGGING_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 100 \
    --load ${LOAD_PATH} \
    --save ${SAVE_PATH} \
    --no-load-optim \
    --no-load-rng
"

torchrun  $DISTRIBUTED_ARGS pretrain_gpt.py \
    $MODEL_ARGS \
    $MOE_ARGS \
    $DATA_ARGS \
    $TRAINING_ARGS \
    $MODEL_PARALLEL_ARGS \
    $LOGGING_ARGS | tee logs/pretrain_grok1_mcore_40b.log
