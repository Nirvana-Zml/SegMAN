#!/usr/bin/env bash
# E2E eval on Trans10K val (default: 100 images; set MAX_IMAGES=-1 for full val)
set -euo pipefail
cd "$(dirname "$0")/.."

MAX_IMAGES="${MAX_IMAGES:-100}"
OUT=outputs/e2e_segment_classify/val_${MAX_IMAGES}

python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val \
  --max-images "${MAX_IMAGES}" \
  --out-dir "${OUT}" \
  --iou-match 0.3 \
  --device cuda:0

echo "Eval done -> ${OUT}/summary.json"
