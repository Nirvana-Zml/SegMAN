#!/usr/bin/env bash
# TransGrasp V1.0 — mode B val eval (M2F + deliver classifier)
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=outputs/e2e_improve/deliver_grasp_e2e_v1
mkdir -p "$(dirname "${OUT}")"

source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate segman_mmdet 2>/dev/null || conda activate segman 2>/dev/null || true

python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --instance-source m2f \
  --m2f-config segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py \
  --m2f-checkpoint segmentation/outputs/m2f_trans10k_pseudo/iter_40000.pth \
  --m2f-score-thresh 0.30 \
  --cls-checkpoint outputs/openclip_classifier/deliver_classifier_best.pth \
  --class-thresholds transgrasp/classification/configs/reject_thresholds_p3.json \
  --out-dir "${OUT}" \
  --min-area 128 --nms-iou 0.5 --iou-match 0.25

conda activate segman 2>/dev/null || true
python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${OUT}"
echo "Mode B done -> ${OUT}"
