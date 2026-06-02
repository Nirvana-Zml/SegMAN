#!/usr/bin/env bash
# E4 regression: full val E2E with optional E1 postprocess params
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-outputs/e2e_segment_classify/regression_$(date +%Y%m%d)}"
SEG_CKPT="${SEG_CKPT:-segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth}"
CLS_CKPT="${CLS_CKPT:-outputs/openclip_classifier/deliver_classifier_best.pth}"

EXTRA_ARGS=("$@")
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
  # Default: best fair E1 from improve plan (override via args)
  EXTRA_ARGS=(--min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 --iou-match 0.25 --min-area-shelf 32)
fi

python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --seg-checkpoint "${SEG_CKPT}" \
  --cls-checkpoint "${CLS_CKPT}" \
  --out-dir "${OUT}" \
  "${EXTRA_ARGS[@]}"

python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${OUT}"
python transgrasp/pipelines/check_e1_gates.py --eval-dir "${OUT}"
echo "Regression -> ${OUT}/summary.json"
