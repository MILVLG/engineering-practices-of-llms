#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 单机8卡配置，如需多机请调整以下参数
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6003
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# 请填写以下路径配置
CKPT_LOAD_DIR="/data3/ep2/models/Qwen/Qwen1.5-7B-mcore"        # 预训练权重（mcore版）路径，例如 ./models/Qwen1.5-7B-mcore
CKPT_SAVE_DIR="/data3/ep2/models/Qwen/Qwen1.5-7B-mcore-lora"   # 微调后权重保存路径
DATA_PATH="/data3/ep2/datasets/alpaca"                  # 指令微调数据（bin/idx 或 HF 指令数据）路径前缀
TOKENIZER_PATH="/data3/ep2/models/Qwen/Qwen1.5-7B"        # 分词器路径，例如 ./models/Qwen1.5-7B/

# TensorBoard 日志目录（可按需修改）
TENSORBOARD_DIR="/data3/ep2/logs/qwen15_7b_full"

# 并行策略（7B推荐 TP=8, PP=1）
TP=8
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --stage sft \
    --finetune \
    --is-instruction-dataset \
    --prompt-type qwen \
    --tokenizer-not-use-fast \
    --variable-seq-lengths \
    --use-mcore-models \
    --use-distributed-optimizer \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 2 \
    --global-batch-size 128 \
    --make-vocab-size-divisible-by 16 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --lr 1.25e-6 \
    --min-lr 1.25e-7 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 0.0 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --add-qkv-bias \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-log-interval 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-world-size-to-tensorboard \
"

FINETUNE_ARGS="
    --finetune \
    --is-instruction-dataset \
    --prompt-type qwen \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"

mkdir -p ${TENSORBOARD_DIR}
mkdir -p logs

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --save ${CKPT_SAVE_DIR} \
    --load ${CKPT_LOAD_DIR} \
    --distributed-backend nccl \
    | tee logs/tune_mcore_qwen15_7b_lora_5000.log
