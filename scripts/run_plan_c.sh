#!/usr/bin/env bash
# Scheme C full pipeline: patch bank -> train -> E2E eval
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segman

bash scripts/run_e2_copypaste_train.sh 2>&1 | tee outputs/e2e_improve/c_train.log

# pick best or latest checkpoint
CKPT_DIR=segmentation/outputs/trans10k_lass_mmscope_balanced_v2_e2copypaste
if compgen -G "${CKPT_DIR}/best_mIoU_iter_"*.pth > /dev/null; then
  SEG_CKPT=$(ls -t "${CKPT_DIR}"/best_mIoU_iter_*.pth | head -1)
else
  SEG_CKPT="${CKPT_DIR}/latest.pth"
fi

SEG_CKPT="${SEG_CKPT}" OUT=outputs/e2e_improve/c_seg_eval \
  bash scripts/run_e2e_c_eval.sh 2>&1 | tee outputs/e2e_improve/c_eval.log

echo "Scheme C complete. See outputs/e2e_improve/c_plan_summary.json"
