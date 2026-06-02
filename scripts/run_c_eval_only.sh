#!/usr/bin/env bash
# Scheme C5 only: E2E eval with existing e2copypaste checkpoint (skip training)
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segman

CKPT_DIR=segmentation/outputs/trans10k_lass_mmscope_balanced_v2_e2copypaste
if [ -n "${SEG_CKPT:-}" ]; then
  :
elif compgen -G "${CKPT_DIR}/best_mIoU_iter_"*.pth > /dev/null; then
  SEG_CKPT=$(ls -t "${CKPT_DIR}"/best_mIoU_iter_*.pth | head -1)
else
  SEG_CKPT="${CKPT_DIR}/iter_2000.pth"
fi

echo "Using checkpoint: ${SEG_CKPT}"
SEG_CKPT="${SEG_CKPT}" OUT="${OUT:-outputs/e2e_improve/c_seg_eval}" \
  bash scripts/run_e2e_c_eval.sh 2>&1 | tee outputs/e2e_improve/c_eval.log

echo "Scheme C eval complete. See outputs/e2e_improve/c_plan_summary.json"
