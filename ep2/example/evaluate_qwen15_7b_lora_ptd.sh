#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CHECKPOINT="/data3/ep2/models/Qwen/Qwen1.5-7B-mcore"           # base 模型 mcore 权重
LORA_CHECKPOINT="/data3/ep2/models/Qwen/Qwen1.5-7B-mcore-lora" # LoRA 适配器权重目录
TOKENIZER_PATH="/data3/ep2/models/Qwen/Qwen1.5-7B"             # 分词器路径（HF 格式）
DATA_PATH="/data3/ep2/engineering-practices-of-llms/ep2/ceval/val" # CEVAL 验证集路径
TASK="ceval"

TP=8
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Optional: TensorBoard logging for evaluation
TENSORBOARD_DIR="/data3/ep2/MindSpeed-LLM/examples/mcore/qwen15/tensorboard/qwen15_7b_lora_eval"
mkdir -p $(dirname "$TENSORBOARD_DIR") "$TENSORBOARD_DIR" 2>/dev/null

torchrun $DISTRIBUTED_ARGS evaluation.py \
       --use-mcore-models \
       --task-data-path $DATA_PATH \
       --task ${TASK} \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --seq-length 8192 \
       --max-new-tokens 1 \
       --max-position-embeddings 8192 \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --num-attention-heads 32  \
       --disable-bias-linear \
       --swiglu \
       --position-embedding-type rope \
       --load $CHECKPOINT \
       --normalization RMSNorm \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --micro-batch-size 1  \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --untie-embeddings-and-output-weights \
       --add-qkv-bias \
       --make-vocab-size-divisible-by 16 \
       --padded-vocab-size 151936 \
       --rotary-base 1000000 \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --bf16 \
       --no-chat-template \
       --lora-r 16 \
       --lora-alpha 32 \
       --lora-fusion \
       --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
       --lora-load ${LORA_CHECKPOINT} \
       --tensorboard-dir ${TENSORBOARD_DIR} \
       --tensorboard-log-interval 1 \
       --log-timers-to-tensorboard \
       --log-batch-size-to-tensorboard \
       --log-memory-to-tensorboard \
       --log-world-size-to-tensorboard \
       | tee logs/eval_mcore_qwen15_7b_lora_${TASK}.log

echo "CEVAL LoRA evaluation finished. TensorBoard dir: ${TENSORBOARD_DIR}"