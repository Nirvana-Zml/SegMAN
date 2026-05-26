#!/usr/bin/env bash
# P3+P2 cascade eval: P2 router + structure head, P3 object head (no overwrite of P2/P3 results)
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE1="${STAGE1:-outputs/openclip_classifier/p2_stage1_router/best.pth}"
STAGE2="${STAGE2:-outputs/openclip_classifier/p2_stage2_structure/best.pth}"
OBJECT="${OBJECT:-outputs/openclip_classifier/p3_p1_hardmining/best.pth}"
GT_REPORT="${GT_REPORT:-outputs/openclip_classifier/p3p2_eval_gt_roi}"
SEG_REPORT="${SEG_REPORT:-outputs/openclip_classifier/p3p2_eval_segman_roi}"

mkdir -p "${GT_REPORT}" "${SEG_REPORT}"

echo "=== P3+P2 GT-ROI eval ==="
echo "  stage1=${STAGE1}"
echo "  stage2=${STAGE2}"
echo "  object=${OBJECT}"
python transgrasp/classification/eval_hierarchical_classifier.py \
  --stage1 "${STAGE1}" \
  --stage2-structure "${STAGE2}" \
  --object-head "${OBJECT}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --report-dir "${GT_REPORT}"

echo "=== P3+P2 SegMAN-ROI eval ==="
python transgrasp/classification/eval_hierarchical_classifier.py \
  --stage1 "${STAGE1}" \
  --stage2-structure "${STAGE2}" \
  --object-head "${OBJECT}" \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir "${SEG_REPORT}"

python transgrasp/classification/tools/summarize_confusion_pairs.py \
  --confusion "${GT_REPORT}/confusion_matrix.json" \
  --topk 15 \
  --out "${GT_REPORT}/top_pairs.json"

python - <<'PY'
import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

gt = {
    'P1单头': 'outputs/openclip_classifier/p1_unfreeze4_noweight/eval_gt_roi/summary.json',
    'P2级联': 'outputs/openclip_classifier/p2_eval_gt_roi/summary.json',
    'P3单头': 'outputs/openclip_classifier/p3_p1_hardmining/eval_gt_roi/summary.json',
    'P3+P2': 'outputs/openclip_classifier/p3p2_eval_gt_roi/summary.json',
}
seg = {
    'P1单头': 'outputs/openclip_classifier/p1_unfreeze4_noweight/eval_segman_roi/summary.json',
    'P2级联': 'outputs/openclip_classifier/p2_eval_segman_roi/summary.json',
    'P3单头': 'outputs/openclip_classifier/p3_p1_hardmining/eval_segman_roi/summary.json',
    'P3+P2': 'outputs/openclip_classifier/p3p2_eval_segman_roi/summary.json',
}

print('=== GT-ROI ===')
for name, p in gt.items():
    s = load(p)
    print(
        f"{name:8s}  acc={s['top1_acc']:.4f}  macro_f1={s['macro_f1']:.4f}  "
        f"door={s['per_class']['door']['f1']:.4f}  wall={s['per_class']['wall']['f1']:.4f}")

print('\n=== SegMAN-ROI ===')
for name, p in seg.items():
    s = load(p)
    print(f"{name:8s}  acc={s['top1_acc']:.4f}  macro_f1={s['macro_f1']:.4f}")

p3p2_gt = load(gt['P3+P2'])
p3p2_seg = load(seg['P3+P2'])
p2_gt = load(gt['P2级联'])
p3_gt = load(gt['P3单头'])
p2_seg = load(seg['P2级联'])
p3_seg = load(seg['P3单头'])

gate = {
    'gt_acc': p3p2_gt['top1_acc'],
    'gt_macro_f1': p3p2_gt['macro_f1'],
    'segman_acc': p3p2_seg['top1_acc'],
    'door_f1': p3p2_gt['per_class']['door']['f1'],
    'wall_f1': p3p2_gt['per_class']['wall']['f1'],
    'vs_p2_gt': round(p3p2_gt['top1_acc'] - p2_gt['top1_acc'], 4),
    'vs_p3_single_gt': round(p3p2_gt['top1_acc'] - p3_gt['top1_acc'], 4),
    'vs_p2_segman': round(p3p2_seg['top1_acc'] - p2_seg['top1_acc'], 4),
    'vs_p3_single_segman': round(p3p2_seg['top1_acc'] - p3_seg['top1_acc'], 4),
    'pass_gt_77': p3p2_gt['top1_acc'] >= 0.77,
    'pass_beat_p3_single': p3p2_gt['top1_acc'] >= p3_gt['top1_acc'],
    'pass_beat_p2_cascade': p3p2_gt['top1_acc'] >= p2_gt['top1_acc'],
}
out = Path('outputs/openclip_classifier/p3p2_eval_gt_roi/gate.json')
out.write_text(json.dumps(gate, indent=2) + '\n', encoding='utf-8')
print('\n--- P3+P2 gate ---')
print(json.dumps(gate, indent=2))
verdict = 'PASS (beat P3 single)' if gate['pass_beat_p3_single'] else (
    'PARTIAL (beat P2 only)' if gate['pass_beat_p2_cascade'] else 'FAIL')
print(verdict)
PY

echo "P3+P2 cascade eval done."
