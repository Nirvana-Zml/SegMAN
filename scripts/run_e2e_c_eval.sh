#!/usr/bin/env bash
# Scheme C5: E2E eval with new seg checkpoint + B1 postprocess
set -euo pipefail
cd "$(dirname "$0")/.."

SEG_CKPT="${SEG_CKPT:-segmentation/outputs/trans10k_lass_mmscope_balanced_v2_e2copypaste/best_mIoU_iter_2000.pth}"
OUT="${OUT:-outputs/e2e_improve/c_seg_eval}"

if [ ! -f "${SEG_CKPT}" ]; then
  # fallback latest
  ALT="segmentation/outputs/trans10k_lass_mmscope_balanced_v2_e2copypaste/latest.pth"
  if [ -f "${ALT}" ]; then
    SEG_CKPT="${ALT}"
  else
    echo "Seg checkpoint not found: ${SEG_CKPT}" >&2
    exit 1
  fi
fi

echo "SEG_CKPT=${SEG_CKPT}"
echo "OUT=${OUT}"

python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --seg-checkpoint "${SEG_CKPT}" \
  --out-dir "${OUT}" \
  --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 \
  --iou-match 0.25 --min-area-shelf 32 \
  --match-algorithm greedy

python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${OUT}"
python transgrasp/pipelines/check_e1_gates.py --eval-dir "${OUT}"

python transgrasp/pipelines/export_unmatched_instances.py \
  --eval-dir "${OUT}" \
  --out-dir outputs/e2e_improve/e2_audit_c_copypaste \
  --sample-wall 100 --sample-door 50

SEG_CKPT="${SEG_CKPT}" OUT="${OUT}" python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ['OUT'])
seg_ckpt = os.environ['SEG_CKPT']
summ = json.loads((out_dir / 'summary.json').read_text(encoding='utf-8'))['aggregate']
gates = {
    'match_ge_65': summ['match_rate'] >= 0.65,
    'pred_gt_le_108': summ['pred_gt_ratio'] <= 1.08,
    'strict_e2e_ge_55': summ['strict_e2e_all_gt'] >= 0.55,
}
c_pass = gates['match_ge_65'] and gates['pred_gt_le_108'] and gates['strict_e2e_ge_55']
ledger = {
    'seg_checkpoint': seg_ckpt,
    'eval_dir': str(out_dir),
    'metrics': summ,
    'gates': gates,
    'C_PASS': c_pass,
    'baseline_match': 0.5932,
    'b1_match': 0.5916,
}
Path('outputs/e2e_improve/c_plan_summary.json').write_text(
    json.dumps(ledger, indent=2) + '\n', encoding='utf-8')
print(json.dumps(ledger, indent=2))
PY

echo "C5 eval -> ${OUT}/summary.json"
