#!/usr/bin/env bash
# Balanced Route B: 10k finetune from baseline. Run inside Docker (cwd: segmentation).
set -euo pipefail

MODE="${1:-lass}"  # lass | dec
WORK_ROOT="${WORK_ROOT:-outputs}"
BASE_CKPT="${BASE_CKPT:-outputs/trans10k_segman_b/iter_80000.pth}"

if [[ "$MODE" == "lass" ]]; then
  CFG="local_configs/segman_trans/segman_b_trans10k_lass_balanced.py"
  WORK_DIR="${WORK_ROOT}/trans10k_lass_mmscope_balanced10k"
else
  CFG="local_configs/segman_trans/segman_b_trans10k_mmscope_balanced.py"
  WORK_DIR="${WORK_ROOT}/trans10k_mmscope_balanced10k"
fi

mkdir -p "$WORK_DIR"
python tools/train.py "$CFG" \
  --work-dir "$WORK_DIR" \
  --load-from "$BASE_CKPT" \
  --no-validate \
  --cfg-options data.workers_per_gpu=2

echo "Train done. Sweep checkpoints:"
for ck in "$WORK_DIR"/iter_*.pth; do
  echo "=== $ck ==="
  python tools/test.py "$CFG" --checkpoint "$ck" --eval mIoU \
    --work-dir "$WORK_DIR/eval_$(basename "$ck" .pth)"
done
