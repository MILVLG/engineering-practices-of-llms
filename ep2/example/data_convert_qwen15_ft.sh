# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh


python ./preprocess_data.py \
      --input /data3/ep2/engineering-practices-of-llms/ep2/alpaca-chinese-52k-v3.json \
      --tokenizer-name-or-path /data3/ep2/models/Qwen/Qwen1.5-7B \
      --output-prefix /data3/ep2/datasets/alpaca \
      --workers 4 \
      --log-interval 1000 \
      --tokenizer-type PretrainedFromHF \
      --handler-name AlpacaStyleInstructionHandler \
      --prompt-type qwen \
      --map-keys '{"prompt": "zh_instruction", "query": "zh_input", "response": "zh_output"}'