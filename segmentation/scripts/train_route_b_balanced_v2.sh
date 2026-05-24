#!/usr/bin/env bash
# balanced-v2: 8k finetune, bowl-oriented losses. fix5k code/config untouched.
# Usage (inside Docker, cwd=segmentation):
#   bash scripts/train_route_b_balanced_v2.sh              # from baseline 80k
#   bash scripts/train_route_b_balanced_v2.sh two_phase   # from fix5k iter_5000
#   nohup bash scripts/train_route_b_balanced_v2.sh > outputs/trans10k_lass_mmscope_balanced_v2/train.log 2>&1 &
set -euo pipefail

CFG="local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py"
WORK_DIR="${WORK_DIR:-outputs/trans10k_lass_mmscope_balanced_v2}"
BASE_CKPT="${BASE_CKPT:-outputs/trans10k_segman_b/iter_80000.pth}"
FIX5K_CKPT="${FIX5K_CKPT:-outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth}"

if [[ "${1:-}" == "two_phase" ]]; then
  LOAD_FROM="$FIX5K_CKPT"
  echo "balanced-v2: init from fix5k ($LOAD_FROM)"
else
  LOAD_FROM="$BASE_CKPT"
  echo "balanced-v2: init from baseline ($LOAD_FROM)"
fi

mkdir -p "$WORK_DIR"
python tools/train.py "$CFG" \
  --work-dir "$WORK_DIR" \
  --load-from "$LOAD_FROM" \
  --no-validate \
  --cfg-options data.workers_per_gpu=2

echo "Train done. Eval checkpoints (mIoU + watch bowl in log):"
for it in 2000 4000 6000 8000; do
  ck="$WORK_DIR/iter_${it}.pth"
  if [[ -f "$ck" ]]; then
    echo "=== $ck ==="
    python tools/test.py "$CFG" --checkpoint "$ck" --eval mIoU \
      --work-dir "$WORK_DIR/eval_iter_${it}"
  fi
done
