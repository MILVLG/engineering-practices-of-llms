# Usage: bash scripts/pt_data_sample.sh

SAVE_DIR=/data2/datasets/pretrain_subset_50GB

if [ -d "$SAVE_DIR" ]; then
    rm -rf $SAVE_DIR
fi

start_time=$(date +%s)

python scripts/data_sample_tool.py \
    --input-root /data2/datasets/Ultra-FineWeb-ShaoZW_subset/ \
    --out-dir $SAVE_DIR \
    --train-size-gb 50 \
    --zh-ratio 0.2 \
    --en-ratio 0.8 \
    --min-score-zh 0.5 \
    --min-score-en 0.5 \
    --target-shard-mb 500

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Data sampling completed in $(expr $elapsed / 60) minutes."