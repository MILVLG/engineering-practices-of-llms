#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH="/data2/datasets/pretrain_preprocessed/train/mmap_data"
VAL_DATA_PATH="/data2/datasets/pretrain_preprocessed/val/mmap_data"
TOKENIZER_MODEL="assets/zen_tokenizer"
TP=1
PP=1
CP=1
TOKENS_PER_STEP=524288
TRAIN_ITERS=20000
BATCH_LENGTH=4096
BATCH_SIZE_PER_DEVICE=4
CKPT_SAVE_DIR="outputs/zen_500m_$(expr $TRAIN_ITERS/1000)k"

echo "=================== Training Config ==============================================="
echo "- Data paths: train=$DATA_PATH,"
echo "                val=$VAL_DATA_PATH"
echo "- Parallel settings: TP=$TP, PP=$PP, CP=$CP, DP=$(($GPUS_PER_NODE/($TP*$PP*$CP))), TP*PP*CP*DP=total $GPUS_PER_NODE GPUs"
echo "- Traing tokens: tokens per step (TPS)=$TOKENS_PER_STEP, train steps (T)=$TRAIN_ITERS,"
echo "                TPS*T=total $(($TOKENS_PER_STEP*$TRAIN_ITERS/1000/1000/1000))B tokens for training"
echo "- Batch settings: length (L)=$BATCH_LENGTH, global batch size (GBS)=TPS/L=$(($TOKENS_PER_STEP / $BATCH_LENGTH)),"
echo "                 batch size per device (B)=$BATCH_SIZE_PER_DEVICE, batch size per forward pass (BFP)=B*DP=$(($BATCH_SIZE_PER_DEVICE * $GPUS_PER_NODE/($TP*$PP*$CP)))," 
echo "                 number of gradient accumulation steps (GAS)=GBS/BFP=$(($TOKENS_PER_STEP / $BATCH_LENGTH / $BATCH_SIZE_PER_DEVICE / $(($GPUS_PER_NODE/($TP*$PP*$CP)))))"
echo "- Checkpoint save directory: $CKPT_SAVE_DIR"
echo "=================== Training Config ================================================"


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
    --no-load-optim \
    --no-load-rng \
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

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --jit-compile \
    --save $CKPT_SAVE_DIR \
    | tee logs/zen_500m_$(expr $TRAIN_ITERS/1000)k_$(date +%Y%m%d_%H%M%S).log
