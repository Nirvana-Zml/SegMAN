#!/usr/bin/env bash
# E2-1 segmentation finetune (run AFTER e2_action_decision.md review)
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segman

CFG=local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2_e2weak.py
OUT=outputs/trans10k_lass_mmscope_balanced_v2_e2weak

cd segmentation
python tools/train.py "${CFG}" --work-dir "${OUT}"

echo "E2-1 training done. Run E2-2 E2E eval with:"
echo "  SEG_CKPT=segmentation/${OUT}/best_mIoU_iter_*.pth bash scripts/run_e2e_e2_eval.sh"
