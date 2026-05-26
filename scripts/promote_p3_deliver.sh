#!/usr/bin/env bash
# Promote P3 checkpoint to formal deliver (archive previous T2 deliver).
set -euo pipefail
cd "$(dirname "$0")/.."

OC=outputs/openclip_classifier
P3="${OC}/p3_p1_hardmining/best.pth"
T2_ARCH="${OC}/deliver_classifier_t2_archived.pth"
DELIVER="${OC}/deliver_classifier_best.pth"
DELIVER_DIR="${OC}/deliver_p3"

if [[ ! -f "${P3}" ]]; then
  echo "Error: missing ${P3}" >&2
  exit 1
fi

mkdir -p "${DELIVER_DIR}/eval_gt_roi"
mkdir -p "${DELIVER_DIR}/eval_segman_roi"

# Archive current T2 deliver alias if it exists and differs from P3
if [[ -f "${DELIVER}" ]]; then
  if ! cmp -s "${DELIVER}" "${P3}" 2>/dev/null; then
    cp -f "${DELIVER}" "${T2_ARCH}"
    echo "Archived previous deliver -> ${T2_ARCH}"
  fi
fi

cp -f "${P3}" "${DELIVER}"
echo "Promoted P3 -> ${DELIVER}"

# Copy eval reports from P3 run
cp -f "${OC}/p3_p1_hardmining/eval_gt_roi/summary.json" "${DELIVER_DIR}/eval_gt_roi/summary.json"
cp -f "${OC}/p3_p1_hardmining/eval_gt_roi/per_class_report.json" "${DELIVER_DIR}/eval_gt_roi/per_class_report.json"
cp -f "${OC}/p3_p1_hardmining/eval_gt_roi/confusion_matrix.json" "${DELIVER_DIR}/eval_gt_roi/confusion_matrix.json"
cp -f "${OC}/p3_p1_hardmining/eval_gt_roi/eval_meta.json" "${DELIVER_DIR}/eval_gt_roi/eval_meta.json" 2>/dev/null || true

cp -f "${OC}/p3_p1_hardmining/eval_segman_roi/summary.json" "${DELIVER_DIR}/eval_segman_roi/summary.json"
cp -f "${OC}/p3_p1_hardmining/eval_segman_roi/per_class_report.json" "${DELIVER_DIR}/eval_segman_roi/per_class_report.json"
cp -f "${OC}/p3_p1_hardmining/eval_segman_roi/confusion_matrix.json" "${DELIVER_DIR}/eval_segman_roi/confusion_matrix.json"
cp -f "${OC}/p3_p1_hardmining/eval_segman_roi/eval_meta.json" "${DELIVER_DIR}/eval_segman_roi/eval_meta.json" 2>/dev/null || true

cp -f "${OC}/p3_p1_hardmining/eval_gt_roi/gate.json" "${DELIVER_DIR}/p3_train_gate.json" 2>/dev/null || true

python - <<'PY'
import json
import shutil
from datetime import date
from pathlib import Path

oc = Path('outputs/openclip_classifier')
gt = json.loads((oc / 'p3_p1_hardmining/eval_gt_roi/summary.json').read_text(encoding='utf-8'))
seg = json.loads((oc / 'p3_p1_hardmining/eval_segman_roi/summary.json').read_text(encoding='utf-8'))
t2_gt = 0.7488
t2_seg = 0.6461

manifest = {
    'archived_at': str(date.today()),
    'status': 'current_deliver',
    'method': 'P3: P1 encoder + hard mining 2x + ColorJitter/CutMix, unfreeze 4 blocks',
    'source': 'outputs/openclip_classifier/p3_p1_hardmining/best.pth',
    'alias': 'outputs/openclip_classifier/deliver_classifier_best.pth',
    'resume_chain': [
        'outputs/openclip_classifier/deliver_classifier_t2_archived.pth',
        'outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth',
        'outputs/openclip_classifier/p3_p1_hardmining/best.pth',
    ],
    'previous_deliver': {
        'method': 'T2 unfreeze 2 blocks',
        'archive_path': 'outputs/openclip_classifier/deliver_classifier_t2_archived.pth',
        'manifest': 'outputs/openclip_classifier/deliver_t2_best/deliver_manifest.json',
        'gt_roi_val': t2_gt,
        'segman_roi_val': t2_seg,
    },
    'segmentation': 'segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth',
    'roi_dataset_train': 'data/trans10k_roi_gt_p3 (weighted sampler; val symlink to gt)',
    'roi_dataset_eval_gt': 'data/trans10k_roi_gt',
    'roi_dataset_eval_segman': 'data/trans10k_roi_segman',
    'metrics': {
        'gt_roi_val': {
            'num_samples': gt['num_samples'],
            'top1_acc': gt['top1_acc'],
            'macro_f1': gt['macro_f1'],
        },
        'segman_roi_val': {
            'num_samples': seg['num_samples'],
            'top1_acc': seg['top1_acc'],
            'macro_f1': seg['macro_f1'],
        },
        'delta_acc_gt_minus_segman': round(gt['top1_acc'] - seg['top1_acc'], 4),
        'vs_previous_t2_deliver': {
            'gt_roi_delta': round(gt['top1_acc'] - t2_gt, 4),
            'segman_roi_delta': round(seg['top1_acc'] - t2_seg, 4),
        },
    },
    'notes': [
        'Formal deliver upgraded from T2 to P3 on user request.',
        'Checkpoint includes fine-tuned encoder (last 4 ViT blocks) + linear head.',
        'GT-ROI 80% stretch target not met (76.91%); see optimization doc v2.6.',
        'Previous T2 deliver preserved at deliver_classifier_t2_archived.pth.',
    ],
}
out = oc / 'deliver_p3' / 'deliver_manifest.json'
out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
# keep symlink-style pointer for tools expecting deliver_t2_best path
legacy = oc / 'deliver_t2_best' / 'deliver_manifest.json'
if legacy.is_file():
    shutil.copy2(out, oc / 'deliver_t2_best' / 'deliver_manifest.json.superseded')
print(json.dumps(manifest['metrics'], indent=2))
PY

echo "Wrote ${DELIVER_DIR}/deliver_manifest.json"
echo "P3 deliver promotion done."
