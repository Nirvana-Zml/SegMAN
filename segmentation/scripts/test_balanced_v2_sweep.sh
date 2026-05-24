#!/usr/bin/env bash
# Sweep balanced-v2 checkpoints; print paths for compare_miou_vs_baseline.py
set -euo pipefail
CFG="local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py"
WORK_DIR="${WORK_DIR:-outputs/trans10k_lass_mmscope_balanced_v2}"

for it in 2000 4000 6000 8000; do
  ck="$WORK_DIR/iter_${it}.pth"
  [[ -f "$ck" ]] || continue
  python tools/test.py "$CFG" --checkpoint "$ck" --eval mIoU \
    --work-dir "$WORK_DIR/eval_iter_${it}"
done

echo "Compare vs baseline (latest json per eval dir):"
for d in "$WORK_DIR"/eval_iter_*/; do
  json=$(ls -1 "$d"/eval_single_scale_*.json 2>/dev/null | tail -1)
  [[ -n "$json" ]] || continue
  echo ">>> $json"
  python scripts/compare_miou_vs_baseline.py "$json"
done
