#!/usr/bin/env bash
# Execute E2E improvement plan: E1 sweep -> E2-0 audit (see docs/E2E_性能分析与改进方案.md)
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=outputs/e2e_segment_classify/val_full
IMPROVE=outputs/e2e_improve

run_eval() {
  local name="$1"
  shift
  local out="${IMPROVE}/${name}"
  echo "=== E1 eval: ${name} ==="
  python transgrasp/pipelines/segment_and_classify.py \
    --eval-split val --max-images -1 \
    --out-dir "${out}" \
    "$@"
  python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${out}"
  python transgrasp/pipelines/check_e1_gates.py --eval-dir "${out}"
}

mkdir -p "${IMPROVE}"

# E1 sweep
run_eval e1_001_min128 --min-area 128
run_eval e1_002_nms128 --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10
run_eval e1_003_full --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 --iou-match 0.25 --min-area-shelf 32

# E2-0 audit on baseline val_full (refresh match metadata on subset is optional)
echo "=== E2-0 audit (baseline val_full) ==="
python transgrasp/pipelines/export_unmatched_instances.py \
  --eval-dir "${BASE}" \
  --out-dir "${IMPROVE}/e2_audit_baseline" \
  --sample-wall 100 --sample-door 50

# Pick best E1 by gate
python - <<'PY'
import json
from pathlib import Path

improve = Path('outputs/e2e_improve')
best_name, best_score = None, -1.0
for d in sorted(improve.glob('e1_*')):
    gate = d / 'e1_gate_check.json'
    if not gate.is_file():
        continue
    g = json.loads(gate.read_text(encoding='utf-8'))
    m = g['metrics']
    score = m['match_rate']
    if g['E1_PASS']:
        score += 0.5
    score += m.get('redundancy_drop_rate', 0) * 0.1
    if score > best_score:
        best_score, best_name = score, d.name
    print(f"{d.name}: E1_PASS={g['E1_PASS']} match={m['match_rate']:.4f} pred_gt={m['pred_gt_ratio']:.4f}")

summary = {'best_e1_dir': str(improve / best_name) if best_name else None, 'best_score': best_score}
Path('outputs/e2e_improve/e1_best.json').write_text(json.dumps(summary, indent=2) + '\n')
print('Best E1:', summary)
PY

echo "Plan E1+E2-0 done. See outputs/e2e_improve/"
