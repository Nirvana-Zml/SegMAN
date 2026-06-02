#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SEG_CKPT="${SEG_CKPT:-segmentation/segmentation/outputs/trans10k_lass_mmscope_balanced_v2_e2weak/best_mIoU_iter_1000.pth}"
OUT="${OUT:-outputs/e2e_improve/e2_seg_eval}"

python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --seg-checkpoint "${SEG_CKPT}" \
  --out-dir "${OUT}" \
  --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 --iou-match 0.25 --min-area-shelf 32

python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${OUT}"
echo "E2-2 eval -> ${OUT}/summary.json"
