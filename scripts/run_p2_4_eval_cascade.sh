#!/usr/bin/env bash
# P2-4: hierarchical cascade eval (GT + SegMAN)
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE1="${STAGE1:-outputs/openclip_classifier/p2_stage1_router/best.pth}"
STAGE2="${STAGE2:-outputs/openclip_classifier/p2_stage2_structure/best.pth}"
OBJECT="${OBJECT:-outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth}"

python transgrasp/classification/eval_hierarchical_classifier.py \
  --stage1 "${STAGE1}" \
  --stage2-structure "${STAGE2}" \
  --object-head "${OBJECT}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --report-dir outputs/openclip_classifier/p2_eval_gt_roi

python transgrasp/classification/eval_hierarchical_classifier.py \
  --stage1 "${STAGE1}" \
  --stage2-structure "${STAGE2}" \
  --object-head "${OBJECT}" \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir outputs/openclip_classifier/p2_eval_segman_roi

python - <<'PY'
import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

p1_gt = load('outputs/openclip_classifier/p1_unfreeze4_noweight/eval_gt_roi/summary.json')
p2_gt = load('outputs/openclip_classifier/p2_eval_gt_roi/summary.json')
p1_seg = load('outputs/openclip_classifier/p1_unfreeze4_noweight/eval_segman_roi/summary.json')
p2_seg = load('outputs/openclip_classifier/p2_eval_segman_roi/summary.json')
print('--- P2 vs P1 (GT-ROI) ---')
print(f"P1 acc={p1_gt['top1_acc']:.4f}  P2 acc={p2_gt['top1_acc']:.4f}  delta={p2_gt['top1_acc']-p1_gt['top1_acc']:+.4f}")
print('--- P2 vs P1 (SegMAN-ROI) ---')
print(f"P1 acc={p1_seg['top1_acc']:.4f}  P2 acc={p2_seg['top1_acc']:.4f}  delta={p2_seg['top1_acc']-p1_seg['top1_acc']:+.4f}")
PY

echo "P2-4 done."
