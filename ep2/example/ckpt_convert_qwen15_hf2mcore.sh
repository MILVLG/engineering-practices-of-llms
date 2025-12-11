# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --load-dir /data3/ep2/models/Qwen/Qwen1.5-7B \
    --save-dir /data3/ep2/models/Qwen/Qwen1.5-7B-mcore \
    --tokenizer-model /data3/ep2/models/Qwen/Qwen1.5-7B \
    --use-mcore-models \
    --model-type-hf llama2 \
    --add-qkv-bias  # --num-layer-list 7,8,9,10,11,11,12,12  --params-dtype bf16 --num-layers-per-virtual-pipeline-stage 2 等参数根据模型需要添加

