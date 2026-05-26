#!/usr/bin/env bash
# P4 quick validation: WiSE-FT sweep + small contrastive adapt -> gate 78%
set -euo pipefail
cd "$(dirname "$0")/.."

P3=outputs/openclip_classifier/p3_p1_hardmining/best.pth
BASELINE_ACC=0.7691
GATE=0.78

mkdir -p outputs/openclip_classifier/p4_wise_ft_sweep
mkdir -p outputs/openclip_classifier/p4_contrastive_small

echo "=== P4-1 WiSE-FT alpha sweep (P3 encoder vs LAION) ==="
python transgrasp/classification/eval_wise_ft_sweep.py \
  --checkpoint "${P3}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --alphas "0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0" \
  --report-dir outputs/openclip_classifier/p4_wise_ft_sweep \
  --save-best \
  2>&1 | tee outputs/openclip_classifier/p4_wise_ft_sweep/sweep.log

WISE_BEST=$(python - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('outputs/openclip_classifier/p4_wise_ft_sweep/sweep.json').read_text())
print(d['best']['top1_acc'])
PY
)

echo "=== P4-2 small-scale contrastive adapt ==="
python transgrasp/classification/train_contrastive_adapt.py \
  --resume "${P3}" \
  --roi-root data/trans10k_roi_gt \
  --work-dir outputs/openclip_classifier/p4_contrastive_small \
  --epochs 4 \
  --batch-size 64 \
  --max-train-samples 8000 \
  --head-finetune-epochs 2 \
  2>&1 | tee outputs/openclip_classifier/p4_contrastive_small/train.log

echo "=== P4-3 eval contrastive best (GT + SegMAN) ==="
CL_BEST=outputs/openclip_classifier/p4_contrastive_small/best.pth
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint "${CL_BEST}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --report-dir outputs/openclip_classifier/p4_contrastive_small/eval_gt_roi

python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint "${CL_BEST}" \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir outputs/openclip_classifier/p4_contrastive_small/eval_segman_roi

python - <<PY
import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

p3 = load('outputs/openclip_classifier/p3_p1_hardmining/eval_gt_roi/summary.json')
wise = load('outputs/openclip_classifier/p4_wise_ft_sweep/sweep.json')
cl = load('outputs/openclip_classifier/p4_contrastive_small/eval_gt_roi/summary.json')
cl_s = load('outputs/openclip_classifier/p4_contrastive_small/eval_segman_roi/summary.json')

gate = {
    'p3_baseline_gt': p3['top1_acc'],
    'gate_78': ${GATE},
    'wise_ft_best_alpha': wise['best']['alpha'],
    'wise_ft_best_gt': wise['best']['top1_acc'],
    'contrastive_gt': cl['top1_acc'],
    'contrastive_segman': cl_s['top1_acc'],
    'wise_pass_78': wise['best']['top1_acc'] >= ${GATE},
    'cl_pass_78': cl['top1_acc'] >= ${GATE},
    'wise_vs_p3': round(wise['best']['top1_acc'] - p3['top1_acc'], 4),
    'cl_vs_p3': round(cl['top1_acc'] - p3['top1_acc'], 4),
}
out = Path('outputs/openclip_classifier/p4_validate_gate.json')
out.write_text(json.dumps(gate, indent=2) + '\n', encoding='utf-8')
print('=== P4 validation gate ===')
print(json.dumps(gate, indent=2))
if gate['wise_pass_78'] or gate['cl_pass_78']:
    print('PASS: at least one method reached 78% GT-ROI')
else:
    print('FAIL: neither method reached 78%; proceed to full P4 or accept P3 best')
PY

echo "P4 validation done."
