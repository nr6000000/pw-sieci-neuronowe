#!/bin/sh
set -eu
cd "$(dirname "$0")"

EPOCHS=300
ONLY="all"
OUT="experiment_results/$(date +%Y%m%d_%H%M%S)"

for DATASET in cube activation simple; do
    uv run python proj1/main.py \
        --dataset "$DATASET" \
        --only "$ONLY" \
        --epochs "$EPOCHS" \
        --batch-size 32 \
        --learning-rate 0.02 \
        --seed 42 \
        --count 100 \
        --out "$OUT/$DATASET"
done

printf '\nResults: %s\n' "$OUT"
