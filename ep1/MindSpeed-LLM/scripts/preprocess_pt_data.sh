#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# === User configurable knobs (match design.md v0.1) ===
DATA_ROOT=${DATA_ROOT:-/data2/datasets/pretrain_subset_50GB}
TOKENIZER_DIR=${TOKENIZER_DIR:-assets/zen_tokenizer}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data2/datasets/pretrain_preprocessed}
SEQ_LEN=${SEQ_LEN:-1000000000}
WORKERS=${WORKERS:-32}

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] DATA_ROOT '$DATA_ROOT' 不存在，检查 design.md 中的数据路径." >&2
  exit 1
fi

if [[ ! -f "$TOKENIZER_DIR/tokenizer.json" ]]; then
  cat <<'EOF' >&2
[ERROR] 未在 TOKENIZER_DIR 中找到 tokenizer.json。
请先运行 tools/export_tiktoken_tokenizer.py 将 tokenizer.pkl 导出到 assets/zen_tokenizer 或传入 TOKENIZER_DIR 指向已有的 HF 目录，再重新执行本脚本。
EOF
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/train" "$OUTPUT_ROOT/val"

    # --streaming \
run_preprocess() {
  local split=$1
  local n_subsets=$2
  local input_path="$DATA_ROOT/$split"
  local out_prefix="$OUTPUT_ROOT/$split/mmap_data"

  if [[ ! -e "$input_path" ]]; then
    echo "[WARN] 跳过 split '$split'，路径 '$input_path' 不存在" >&2
    return
  fi

  echo "[INFO] 开始处理 $split -> $out_prefix"
  python ./preprocess_data.py \
    --input "$input_path" \
    --handler-name GeneralPretrainHandler \
    --json-keys content \
    --n-subs "$n_subsets" \
    --cache-dir "$out_prefix.cache" \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path "$TOKENIZER_DIR" \
    --seq-length "$SEQ_LEN" \
    --append-eod \
    --dataset-impl mmap \
    --output-prefix "$out_prefix" \
    --workers "$WORKERS" \
    --log-interval 200
}

run_preprocess train 8
run_preprocess val 1

echo "[INFO] 预处理完成，产物位于 $OUTPUT_ROOT/{train,val}"
