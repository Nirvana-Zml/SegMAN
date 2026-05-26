#!/usr/bin/env bash
# P4F-5/6 eval only (after train completed)
set -euo pipefail
cd "$(dirname "$0")/.."
BEST=outputs/openclip_classifier/p4_contrastive_full/best.pth

python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint "${BEST}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --report-dir outputs/openclip_classifier/p4_contrastive_full/eval_gt_roi

python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint "${BEST}" \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir outputs/openclip_classifier/p4_contrastive_full/eval_segman_roi

python transgrasp/classification/tools/summarize_confusion_pairs.py \
  --confusion outputs/openclip_classifier/p4_contrastive_full/eval_gt_roi/confusion_matrix.json \
  --topk 15 \
  --out outputs/openclip_classifier/p4_contrastive_full/eval_gt_roi/top_pairs.json

python - <<'PY'
import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

p3 = load('outputs/openclip_classifier/p3_p1_hardmining/eval_gt_roi/summary.json')
p3s = load('outputs/openclip_classifier/p3_p1_hardmining/eval_segman_roi/summary.json')
p4s = load('outputs/openclip_classifier/p4_contrastive_small/eval_gt_roi/summary.json')
p4 = load('outputs/openclip_classifier/p4_contrastive_full/eval_gt_roi/summary.json')
p4_seg = load('outputs/openclip_classifier/p4_contrastive_full/eval_segman_roi/summary.json')
wise = load('outputs/openclip_classifier/p4_full_wise_ft/sweep.json')

gate = {
    'p3_gt': p3['top1_acc'],
    'p4_small_gt': p4s['top1_acc'],
    'p4_full_gt': p4['top1_acc'],
    'p4_full_segman': p4_seg['top1_acc'],
    'door_f1': p4['per_class']['door']['f1'],
    'wall_f1': p4['per_class']['wall']['f1'],
    'wise_ft_best_alpha': wise['best']['alpha'],
    'wise_ft_best_gt': wise['best']['top1_acc'],
    'vs_p3_gt': round(p4['top1_acc'] - p3['top1_acc'], 4),
    'vs_p4_small_gt': round(p4['top1_acc'] - p4s['top1_acc'], 4),
    'vs_p3_segman': round(p4_seg['top1_acc'] - p3s['top1_acc'], 4),
    'pass_78': p4['top1_acc'] >= 0.78,
    'pass_80': p4['top1_acc'] >= 0.80,
    'pass_beat_p4_small': p4['top1_acc'] >= p4s['top1_acc'],
    'rollback_p3': p4['top1_acc'] < p3['top1_acc'],
}
Path('outputs/openclip_classifier/p4_contrastive_full/eval_gt_roi/gate.json').write_text(
    json.dumps(gate, indent=2) + '\n', encoding='utf-8')
print(json.dumps(gate, indent=2))
PY
