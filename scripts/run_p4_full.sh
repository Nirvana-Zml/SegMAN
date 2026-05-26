#!/usr/bin/env bash
# P4-full: contrastive (full train) -> WiSE-FT -> GT/SegMAN eval -> gate 78%/80%
set -euo pipefail
cd "$(dirname "$0")/.."

P3=outputs/openclip_classifier/p3_p1_hardmining/best.pth
WORK=outputs/openclip_classifier/p4_contrastive_full
GATE_78=0.78
GATE_80=0.80
P4_SMALL=0.7704

mkdir -p "${WORK}"
mkdir -p outputs/openclip_classifier/p4_full_wise_ft

echo "=== P4F-0 baseline check ==="
test -f "${P3}"
python - <<'PY'
import json
from pathlib import Path
for p in [
    'outputs/openclip_classifier/p3_p1_hardmining/eval_gt_roi/summary.json',
    'outputs/openclip_classifier/p4_validate_gate.json',
]:
    print(p, 'OK' if Path(p).is_file() else 'MISSING')
p3 = json.loads(Path('outputs/openclip_classifier/p3_p1_hardmining/eval_gt_roi/summary.json').read_text())
print(f"P3 baseline GT acc={p3['top1_acc']:.4f}")
PY

echo "=== P4F-2 full contrastive ==="
python transgrasp/classification/train_contrastive_adapt.py \
  --config transgrasp/classification/configs/p4_contrastive_full.yaml \
  2>&1 | tee "${WORK}/train.log"

BEST="${WORK}/best.pth"

echo "=== P4F-4 WiSE-FT on P4-full best ==="
python transgrasp/classification/eval_wise_ft_sweep.py \
  --checkpoint "${BEST}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --alphas "0.9,0.92,0.94,0.95,0.96,0.98,1.0" \
  --report-dir outputs/openclip_classifier/p4_full_wise_ft \
  --save-best

WISE_BEST=outputs/openclip_classifier/p4_full_wise_ft/best.pth
EVAL_CKPT=$(python - <<'PY'
import json
from pathlib import Path

cl = json.loads(Path('outputs/openclip_classifier/p4_contrastive_full/train_summary.json').read_text())['best_val_acc']
ckpt = 'outputs/openclip_classifier/p4_contrastive_full/best.pth'
wise_path = Path('outputs/openclip_classifier/p4_full_wise_ft/sweep.json')
if wise_path.is_file():
    wise = json.loads(wise_path.read_text())['best']['top1_acc']
    if wise > cl:
        ckpt = 'outputs/openclip_classifier/p4_full_wise_ft/best.pth'
print(ckpt, end='')
PY
)
echo "Final eval checkpoint: ${EVAL_CKPT}"

echo "=== P4F-5 GT eval ==="
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint "${EVAL_CKPT}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --report-dir "${WORK}/eval_gt_roi"

echo "=== P4F-5 SegMAN eval ==="
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint "${EVAL_CKPT}" \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir "${WORK}/eval_segman_roi"

python transgrasp/classification/tools/summarize_confusion_pairs.py \
  --confusion "${WORK}/eval_gt_roi/confusion_matrix.json" \
  --topk 15 \
  --out "${WORK}/eval_gt_roi/top_pairs.json"

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
wise = load('outputs/openclip_classifier/p4_full_wise_ft/sweep.json') if Path(
    'outputs/openclip_classifier/p4_full_wise_ft/sweep.json').is_file() else None

gate = {
    'p3_gt': p3['top1_acc'],
    'p4_small_gt': p4s['top1_acc'],
    'p4_full_gt': p4['top1_acc'],
    'p4_full_segman': p4_seg['top1_acc'],
    'door_f1': p4['per_class']['door']['f1'],
    'wall_f1': p4['per_class']['wall']['f1'],
    'vs_p3_gt': round(p4['top1_acc'] - p3['top1_acc'], 4),
    'vs_p4_small_gt': round(p4['top1_acc'] - p4s['top1_acc'], 4),
    'vs_p3_segman': round(p4_seg['top1_acc'] - p3s['top1_acc'], 4),
    'pass_78': p4['top1_acc'] >= 0.78,
    'pass_80': p4['top1_acc'] >= 0.80,
    'pass_beat_p4_small': p4['top1_acc'] >= p4s['top1_acc'],
    'rollback_p3': p4['top1_acc'] < p3['top1_acc'],
}
if wise:
    gate['wise_ft_best_alpha'] = wise['best']['alpha']
    gate['wise_ft_best_gt'] = wise['best']['top1_acc']

out = Path('outputs/openclip_classifier/p4_contrastive_full/eval_gt_roi/gate.json')
out.write_text(json.dumps(gate, indent=2) + '\n', encoding='utf-8')
print('=== P4-full gate ===')
print(json.dumps(gate, indent=2))
if gate['pass_80']:
    print('PASS 80%')
elif gate['pass_78']:
    print('PASS 78% stretch')
elif gate['pass_beat_p4_small']:
    print('PARTIAL: beat P4-small but <78%')
elif gate['rollback_p3']:
    print('FAIL: below P3 — rollback')
else:
    print('FAIL: below 78%')
PY

echo "P4-full done."
