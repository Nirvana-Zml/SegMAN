#!/usr/bin/env bash
# F1 P1: build M2F-matched ROI dataset -> fine-tune classifier -> E2E eval + gate
set -euo pipefail
cd "$(dirname "$0")/.."

source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
if [[ -f /root/anaconda3/etc/profile.d/conda.sh ]]; then
  source /root/anaconda3/etc/profile.d/conda.sh
fi

CKPT=segmentation/outputs/m2f_trans10k_pseudo/iter_40000.pth
M2F_CFG=segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py
ROI_ROOT=data/trans10k_roi_m2f
WORK=outputs/openclip_classifier/m2f_roi_adapt_v1
E2E_OUT=outputs/e2e_improve/f1_m2f_e2e_cls_adapt_v1
LOG=outputs/e2e_improve/f1_p1_adapt.log
mkdir -p outputs/e2e_improve

exec >> "${LOG}" 2>&1
echo "========== F1 P1 start $(date) =========="

build_split() {
  local split="$1"
  echo "========== build M2F ROI ${split} $(date) =========="
  conda activate segman_mmdet
  python transgrasp/pipelines/build_m2f_roi_dataset.py \
    --split "${split}" \
    --out-root "${ROI_ROOT}" \
    --m2f-config "${M2F_CFG}" \
    --m2f-checkpoint "${CKPT}" \
    --m2f-score-thresh 0.30 \
    --min-area 128 --nms-iou 0.5 --iou-match 0.25
}

if [[ ! -f "${ROI_ROOT}/train/labels.csv" ]]; then
  build_split train
else
  echo "Reuse ${ROI_ROOT}/train"
fi
if [[ ! -f "${ROI_ROOT}/val/labels.csv" ]]; then
  build_split val
else
  echo "Reuse ${ROI_ROOT}/val"
fi

echo "========== train classifier $(date) =========="
conda activate segman
python transgrasp/classification/train_openclip_classifier.py \
  --config transgrasp/classification/configs/f1_m2f_roi_adapt_v1.yaml

BEST="${WORK}/best.pth"
if [[ ! -f "${BEST}" ]]; then
  echo "ERROR: missing ${BEST}" >&2
  exit 1
fi

echo "========== E2E eval cls_adapt_v1 $(date) =========="
conda activate segman_mmdet
python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --instance-source m2f \
  --m2f-config "${M2F_CFG}" \
  --m2f-checkpoint "${CKPT}" \
  --m2f-score-thresh 0.30 \
  --cls-checkpoint "${BEST}" \
  --out-dir "${E2E_OUT}" \
  --min-area 128 --nms-iou 0.5 --iou-match 0.25

conda activate segman
python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${E2E_OUT}"
python transgrasp/pipelines/run_f1_gate_check.py

echo "========== F1 P1 DONE $(date) =========="
