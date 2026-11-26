#!/usr/bin/env bash
set -euo pipefail
lesson="$(cd "$(dirname "$0")" && pwd)"
cd "${EP2_WORKDIR:?Set EP2_WORKDIR to your experiment directory}"
mode="$1"
run="$(realpath "$2")"
trap 'code=$?; printf "%s\n" "$code" > "$run/exit-code"' EXIT
export PATH="$PWD/.venv/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Select four GPUs}" NPROC_PER_NODE=4
export HF_HOME="$PWD/.cache/huggingface" HF_DATASETS_CACHE="$PWD/.cache/huggingface/datasets" MODELSCOPE_CACHE="$PWD/.cache/modelscope"
export XDG_CACHE_HOME="$PWD/.cache" TMPDIR="$PWD/.cache/tmp" HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export SWANLAB_LOG_DIR="$run/swanlog" SWANLAB_CONFIG_DIR="$PWD/.cache/swanlab" SWANLAB_SAVE_DIR="$PWD/.cache/swanlab"
case "$mode" in
  smoke)
    shards=("$PWD"/datasets/dolci/data/*.parquet)
    test -f "${shards[0]}"
    datasets=("${shards[0]}#256")
    options=(--max_steps 2 --logging_steps 1 --save_steps 2 --dataset_num_proc 1 --report_to none)
    ;;
  train)
    dataset_dir="$PWD/datasets/dolci/data"
    datasets=("$dataset_dir"/*.parquet)
    test -f "${datasets[0]}"
    if [[ -z "${SWANLAB_API_KEY:-}" && -r .secrets/swanlab.key ]]; then
      SWANLAB_API_KEY="$(< .secrets/swanlab.key)"
    fi
    : "${SWANLAB_API_KEY:?Set the SwanLab API key before training}"
    export SWANLAB_API_KEY
    options=(--logging_steps 10 --save_steps 1000000 --dataset_num_proc 8
             --external_plugins "$lesson/half_epoch_save.py" --callbacks pocket_half_epoch_save
             --report_to swanlab --swanlab_project pocket-dolci-sft
             --swanlab_exp_name "$(basename "$run")" --swanlab_mode cloud)
    ;;
  *) exit 2 ;;
esac
python -c 'import torch; assert torch.cuda.device_count() == 4, "Select four visible CUDA devices"'
swift sft \
    --model "$PWD/models/MNN-Pocket-0.3B-Base" --model_type qwen3 \
    --template qwen3_nothinking --max_length 4096 --attn_impl sdpa \
    --tuner_type full --torch_dtype float32 --fp16 false --bf16 true --num_train_epochs 3 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 --warmup_ratio 0.05 --lr_scheduler_type cosine \
    --dataset "${datasets[@]}" --load_from_cache_file true --split_dataset_ratio 0 --lazy_tokenize true \
    --gradient_checkpointing true --ddp_find_unused_parameters false --dataloader_num_workers 4 \
    --output_dir "$run/checkpoints" --add_version false --save_strategy steps --save_total_limit 6 \
    "${options[@]}"
