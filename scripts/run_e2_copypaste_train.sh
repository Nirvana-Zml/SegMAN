#!/usr/bin/env bash
# Scheme C: build patch bank + finetune segmentation (Copy-Paste)
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segman

echo "=== C1: Build Copy-Paste patch bank ==="
cd segmentation
python tools/build_copypaste_patch_bank.py \
  --data-root data/trans10k \
  --out data/trans10k/copypaste_patch_bank.pkl \
  --max-patches-per-class 400

echo "=== C4: Train e2copypaste (3000 iter) ==="
CFG=local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2_e2copypaste.py
OUT=outputs/trans10k_lass_mmscope_balanced_v2_e2copypaste

python tools/train.py "${CFG}" --work-dir "${OUT}"

echo "Training done. Checkpoint dir: segmentation/${OUT}/"
echo "Run E2E: bash scripts/run_e2e_c_eval.sh"
