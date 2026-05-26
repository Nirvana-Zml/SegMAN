#!/usr/bin/env bash
# P1 two-stage: warmup 2 blocks (save encoder) -> deepen 4 blocks -> eval
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p outputs/openclip_classifier/p1_warmup_unfreeze2
mkdir -p outputs/openclip_classifier/p1_unfreeze4_noweight

echo "=== P1 stage-1: warmup unfreeze 2 ==="
python transgrasp/classification/train_openclip_classifier.py \
  --config transgrasp/classification/configs/p1_warmup_unfreeze2.yaml \
  --no-class-weights \
  2>&1 | tee outputs/openclip_classifier/p1_warmup_unfreeze2/train.log

echo "=== P1 stage-2: unfreeze 4 ==="
python transgrasp/classification/train_openclip_classifier.py \
  --config transgrasp/classification/configs/p1_unfreeze4_noweight.yaml \
  --no-class-weights \
  2>&1 | tee outputs/openclip_classifier/p1_unfreeze4_noweight/train.log

WORK=outputs/openclip_classifier/p1_unfreeze4_noweight
BEST="${WORK}/best.pth"

echo "=== P1 GT-ROI eval ==="
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint "${BEST}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --report-dir "${WORK}/eval_gt_roi"

echo "=== P1 SegMAN-ROI eval ==="
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint "${BEST}" \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir "${WORK}/eval_segman_roi"

python - <<'PY'
import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

t2 = load('outputs/openclip_classifier/t2_unfreeze2_noweight/eval_gt_roi/summary.json')
p1 = load('outputs/openclip_classifier/p1_unfreeze4_noweight/eval_gt_roi/summary.json')
print('--- P1 gate (GT-ROI) ---')
print(f"T2 baseline: acc={t2['top1_acc']:.4f} macro_f1={t2['macro_f1']:.4f}")
print(f"P1 result:   acc={p1['top1_acc']:.4f} macro_f1={p1['macro_f1']:.4f}")
ok = p1['top1_acc'] >= t2['top1_acc'] and p1['macro_f1'] >= t2['macro_f1']
print('PASS' if ok else 'FAIL (keep deliver_classifier_best.pth)')
PY

echo "P1 pipeline done."
