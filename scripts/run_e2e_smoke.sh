#!/usr/bin/env bash
# E2E smoke test: one val image
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=outputs/e2e_segment_classify/smoke
IMG=segmentation/data/trans10k/img_dir/val/val_000000.jpg
if [[ ! -f "${IMG}" ]]; then
  IMG=$(ls segmentation/data/trans10k/img_dir/val/*.jpg 2>/dev/null | head -1)
fi
if [[ -z "${IMG:-}" || ! -f "${IMG}" ]]; then
  echo "No val image found under segmentation/data/trans10k/img_dir/val"
  exit 1
fi

python transgrasp/pipelines/segment_and_classify.py \
  --image "${IMG}" \
  --out-dir "${OUT}" \
  --save-rois \
  --save-sem-seg \
  --device cuda:0

echo "Smoke OK. See ${OUT}/"
