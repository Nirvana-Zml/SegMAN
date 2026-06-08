#!/usr/bin/env bash
# Phase 1: SegMAN semantic track match_rate post-process sweep
# See docs/优化SegMANmatch_rate/SegMAN_match_rate_提升实施计划.md §3.1
set -euo pipefail
cd "$(dirname "$0")/.."

IMPROVE=outputs/match_improve
SEG_CKPT="${SEG_CKPT:-segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth}"
CLS_CKPT="${CLS_CKPT:-outputs/openclip_classifier/deliver_classifier_best.pth}"
MAX_IMAGES="${MAX_IMAGES:--1}"

BASE_ARGS=(
  --instance-source semantic
  --seg-checkpoint "${SEG_CKPT}"
  --cls-checkpoint "${CLS_CKPT}"
  --min-area 128
  --nms-iou 0.5
  --max-aspect-ratio 10
  --iou-match 0.25
  --min-area-shelf 32
)

run_one() {
  local name="$1"
  shift
  local out="${IMPROVE}/${name}"
  echo "=== Phase1: ${name} ==="
  python transgrasp/pipelines/segment_and_classify.py \
    --eval-split val --max-images "${MAX_IMAGES}" \
    --out-dir "${out}" \
    "${BASE_ARGS[@]}" "$@"
  python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${out}"
  python transgrasp/pipelines/check_e1_gates.py --eval-dir "${out}" || true
}

mkdir -p "${IMPROVE}"

if [[ "${1:-}" == "--quick" ]]; then
  MAX_IMAGES=50
  shift
fi

run_one phase1_p0_b1_baseline
run_one phase1_p1_tta_default \
  --seg-tta --seg-tta-scales 0.75,1.0,1.25
run_one phase1_p1_m1_dist_only \
  --seg-tta --seg-tta-scales 0.75,1.0,1.25 \
  --merge-cc-iou 0 --merge-cc-dist 12 --merge-cc-classes wall,door
run_one phase1_p1_m2_conservative \
  --seg-tta --seg-tta-scales 0.75,1.0,1.25 \
  --merge-cc-iou 0.08 --merge-cc-dist 12 --merge-cc-classes wall,door,window
run_one phase1_p1_iou_per_class \
  --seg-tta --seg-tta-scales 0.75,1.0,1.25 \
  --iou-match-per-class door:0.22,wall:0.22,window:0.22,shelf:0.22

python - <<'PY'
import json
from pathlib import Path

improve = Path('outputs/match_improve')
rows = []
for d in sorted(improve.glob('phase1_*')):
    report = d / 'e2e_metrics_report.json'
    if not report.is_file():
        continue
    r = json.loads(report.read_text(encoding='utf-8'))
    il = r['instance_level']
    rows.append({
        'name': d.name,
        'match_rate': il['match_rate'],
        'pred_gt_ratio': il.get('pred_gt_ratio'),
        'cls_on_matched': il.get('e2e_top1_on_matched'),
        'wall_match': r.get('per_class_gt_instance', {}).get('wall', {}).get('match_rate'),
    })

best = max(rows, key=lambda x: x['match_rate']) if rows else None
summary = {'runs': rows, 'best': best}
out = improve / 'phase1_summary.json'
baseline = next((r for r in rows if r['name'] == 'phase1_p0_b1_baseline'), None)
if best and baseline:
    summary['delta_pp'] = round((best['match_rate'] - baseline['match_rate']) * 100, 2)
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2, ensure_ascii=False))
print('Wrote', out)
PY

echo "Phase 1 sweep done -> ${IMPROVE}/phase1_summary.json"
