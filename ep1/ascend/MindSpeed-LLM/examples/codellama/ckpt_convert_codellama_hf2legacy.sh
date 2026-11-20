# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/CodeLlama-hf/ \
    --save-dir ./model_weights/CodeLlama-legacy/ \
    --tokenizer-model ./model_from_hf/CodeLlama-hf/tokenizer.model \
    --params-dtype bf16