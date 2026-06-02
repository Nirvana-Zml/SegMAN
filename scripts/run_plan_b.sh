#!/usr/bin/env bash
# Scheme B: instance pipeline tuning (B1–B5)
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate segman 2>/dev/null || true

IMPROVE=outputs/e2e_improve
mkdir -p "${IMPROVE}"

# Shared B1+B4 pred postprocess
B1_ARGS=(
  --min-area 128
  --nms-iou 0.5
  --max-aspect-ratio 10
  --iou-match 0.25
  --min-area-shelf 32
)

run_eval() {
  local name="$1"
  shift
  echo ""
  echo "========== ${name} =========="
  python transgrasp/pipelines/segment_and_classify.py \
    --eval-split val --max-images -1 \
    --out-dir "${IMPROVE}/${name}" \
    "${B1_ARGS[@]}" \
    "$@"
  python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${IMPROVE}/${name}"
  python transgrasp/pipelines/check_e1_gates.py --eval-dir "${IMPROVE}/${name}"
}

# B1+B4 deploy baseline (greedy, global iou 0.25)
run_eval b_b1_deploy --match-algorithm greedy

# B2: Hungarian vs greedy
run_eval b2_greedy --match-algorithm greedy
run_eval b2_hungarian --match-algorithm hungarian

# B3: per-class IoU thresholds (eval only)
run_eval b3_per_class_iou \
  --match-algorithm greedy \
  --iou-match-per-class "door:0.25,wall:0.25,cup:0.35"

# B5: CC merge wall/door
run_eval b5_no_merge --match-algorithm greedy
run_eval b5_merge_wd \
  --match-algorithm greedy \
  --merge-cc-iou 0.3 \
  --merge-cc-classes wall,door

# Combined: B2 + B5
run_eval b_combined \
  --match-algorithm hungarian \
  --merge-cc-iou 0.3 \
  --merge-cc-classes wall,door

# E2-0 audit on best candidates
for d in b_b1_deploy b2_hungarian b5_merge_wd b_combined; do
  python transgrasp/pipelines/export_unmatched_instances.py \
    --eval-dir "${IMPROVE}/${d}" \
    --out-dir "${IMPROVE}/e2_audit_${d}" \
    --sample-wall 50 --sample-door 30 2>/dev/null || true
done

# Summary ledger
python - <<'PY'
import json
from pathlib import Path

improve = Path('outputs/e2e_improve')
baseline = {'match_rate': 0.5932, 'pred_gt_ratio': 1.148}
rows = []
for name in [
    'b_b1_deploy', 'b2_greedy', 'b2_hungarian', 'b3_per_class_iou',
    'b5_no_merge', 'b5_merge_wd', 'b_combined',
]:
    d = improve / name
    summ = d / 'summary.json'
    gate = d / 'e1_gate_check.json'
    if not summ.is_file():
        continue
    agg = json.loads(summ.read_text(encoding='utf-8')).get('aggregate', {})
    if agg.get('num_gt_instances', 0) != 3105:
        continue
    g = json.loads(gate.read_text(encoding='utf-8')) if gate.is_file() else {}
    rows.append({
        'name': name,
        'match_rate': agg.get('match_rate'),
        'pred_gt_ratio': agg.get('pred_gt_ratio'),
        'strict_e2e': agg.get('strict_e2e_all_gt'),
        'cls_on_matched': agg.get('e2e_top1_on_matched'),
        'E1_PASS': g.get('E1_PASS'),
    })

best = max(rows, key=lambda r: (r['match_rate'], -abs(r['pred_gt_ratio'] - 1.05))) if rows else None
out = {
    'baseline': baseline,
    'runs': rows,
    'best': best,
    'deploy_recommendation': {
        'cli_args': [
            '--min-area', '128', '--nms-iou', '0.5', '--max-aspect-ratio', '10',
            '--iou-match', '0.25', '--min-area-shelf', '32',
        ],
        'note': 'B1 deploy only; do NOT enable merge_cc_iou=0.3 (B5 FAIL)',
    },
}
Path('outputs/e2e_improve/b_plan_summary.json').write_text(
    json.dumps(out, indent=2) + '\n', encoding='utf-8')
print(json.dumps(out, indent=2))
PY

echo ""
echo "Scheme B done. See outputs/e2e_improve/b_plan_summary.json"
