#!/usr/bin/env bash
# P3: audit -> manifest -> train -> eval -> gate
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/openclip_classifier/p3_hard_audit
mkdir -p outputs/openclip_classifier/p3_p1_hardmining

echo "=== P3-0 audit ==="
python transgrasp/classification/tools/summarize_confusion_pairs.py \
  --confusion outputs/openclip_classifier/p1_unfreeze4_noweight/eval_gt_roi/confusion_matrix.json \
  --topk 15 \
  --out outputs/openclip_classifier/p3_hard_audit/top_pairs_p1.json

python transgrasp/classification/tools/summarize_confusion_pairs.py \
  --confusion outputs/openclip_classifier/p2_eval_gt_roi/confusion_matrix.json \
  --topk 15 \
  --out outputs/openclip_classifier/p3_hard_audit/top_pairs_p2.json

echo "=== P3-1 manifest ==="
python transgrasp/data/build_hard_pair_manifest.py \
  --roi-root data/trans10k_roi_gt \
  --out-root data/trans10k_roi_gt_p3 \
  --confusion outputs/openclip_classifier/p1_unfreeze4_noweight/eval_gt_roi/confusion_matrix.json \
  --force

echo "=== P3-3 train ==="
python transgrasp/classification/train_openclip_classifier.py \
  --config transgrasp/classification/configs/p3_p1_hardmining.yaml \
  --no-class-weights \
  2>&1 | tee outputs/openclip_classifier/p3_p1_hardmining/train.log

BEST=outputs/openclip_classifier/p3_p1_hardmining/best.pth

echo "=== P3-4 GT eval ==="
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint "${BEST}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --report-dir outputs/openclip_classifier/p3_p1_hardmining/eval_gt_roi

echo "=== P3-4 SegMAN eval ==="
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint "${BEST}" \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir outputs/openclip_classifier/p3_p1_hardmining/eval_segman_roi

python - <<'PY'
import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

p1 = load('outputs/openclip_classifier/p1_unfreeze4_noweight/eval_gt_roi/summary.json')
p2 = load('outputs/openclip_classifier/p2_eval_gt_roi/summary.json')
p3 = load('outputs/openclip_classifier/p3_p1_hardmining/eval_gt_roi/summary.json')
p3s = load('outputs/openclip_classifier/p3_p1_hardmining/eval_segman_roi/summary.json')
p1s = load('outputs/openclip_classifier/p1_unfreeze4_noweight/eval_segman_roi/summary.json')

gate = {
    'gt_acc': p3['top1_acc'],
    'gt_macro_f1': p3['macro_f1'],
    'segman_acc': p3s['top1_acc'],
    'door_f1': p3['per_class']['door']['f1'],
    'wall_f1': p3['per_class']['wall']['f1'],
    'vs_p1_gt': round(p3['top1_acc'] - p1['top1_acc'], 4),
    'vs_p2_gt': round(p3['top1_acc'] - p2['top1_acc'], 4),
    'vs_p1_segman': round(p3s['top1_acc'] - p1s['top1_acc'], 4),
    'pass_gt_765': p3['top1_acc'] >= 0.765,
    'pass_gt_p1': p3['top1_acc'] >= p1['top1_acc'],
    'pass_door_66': p3['per_class']['door']['f1'] >= 0.66,
    'pass_segman': p3s['top1_acc'] >= 0.655,
}
out = Path('outputs/openclip_classifier/p3_p1_hardmining/eval_gt_roi/gate.json')
out.write_text(json.dumps(gate, indent=2) + '\n', encoding='utf-8')
print('--- P3 gate ---')
print(json.dumps(gate, indent=2))
print('PASS' if gate['pass_gt_p1'] and gate['pass_segman'] else 'FAIL')
PY

echo "P3 done."
