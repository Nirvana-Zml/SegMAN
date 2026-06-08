#!/usr/bin/env bash
# TransGrasp V1.0 — mode A val eval (semantic + deliver classifier)
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=outputs/e2e_improve/deliver_semantic_e2e_v1
mkdir -p "$(dirname "${OUT}")"

source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate segman 2>/dev/null || true

python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --instance-source semantic \
  --seg-config segmentation/local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --cls-checkpoint outputs/openclip_classifier/deliver_classifier_best.pth \
  --class-thresholds transgrasp/classification/configs/reject_thresholds_p3.json \
  --out-dir "${OUT}" \
  --min-area 128 --nms-iou 0.5 --iou-match 0.25

python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${OUT}"
echo "Mode A done -> ${OUT}"
