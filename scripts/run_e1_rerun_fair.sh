#!/usr/bin/env bash
# Re-run E1 with fair GT extraction (GT always min_area=64, no NMS)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/e2e_improve

run_one() {
  local name="$1"
  shift
  echo "=== ${name} ==="
  python transgrasp/pipelines/segment_and_classify.py \
    --eval-split val --max-images -1 \
    --out-dir "outputs/e2e_improve/${name}" \
    "$@"
  python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "outputs/e2e_improve/${name}"
  python transgrasp/pipelines/check_e1_gates.py --eval-dir "outputs/e2e_improve/${name}"
}

# Fair GT baseline comparisons
run_one e1_003_full_fair --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 \
  --iou-match 0.25 --min-area-shelf 32

run_one e1_004_iou028 --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 \
  --iou-match 0.28 --min-area-shelf 32

run_one e1_005_balanced --min-area 96 --nms-iou 0.5 --max-aspect-ratio 10 \
  --iou-match 0.3 --min-area-shelf 48

python transgrasp/pipelines/export_unmatched_instances.py \
  --eval-dir outputs/e2e_improve/e1_003_full_fair \
  --out-dir outputs/e2e_improve/e2_audit_e1_best 2>/dev/null || \
python transgrasp/pipelines/export_unmatched_instances.py \
  --eval-dir outputs/e2e_improve/e1_004_iou028 \
  --out-dir outputs/e2e_improve/e2_audit_e1_best \
  --sample-wall 100 --sample-door 50

python - <<'PY'
import json
from pathlib import Path
improve = Path('outputs/e2e_improve')
rows = []
best_name, best_score = None, -1.0
for d in sorted(improve.glob('e1_*')):
    gate = d / 'e1_gate_check.json'
    summ = d / 'summary.json'
    if not gate.is_file() or not summ.is_file():
        continue
    g = json.loads(gate.read_text(encoding='utf-8'))
    s = json.loads(summ.read_text(encoding='utf-8')).get('aggregate', {})
    if s.get('num_gt_instances', 0) != 3105:
        print('SKIP (unfair GT count):', d.name, s.get('num_gt_instances'))
        continue
    m = g['metrics']
    score = m['match_rate'] + (0.5 if g['E1_PASS'] else 0) + m.get('redundancy_drop_rate', 0) * 0.1
    rows.append((d.name, g['E1_PASS'], m))
    if score > best_score:
        best_score, best_name = score, d.name
for name, passed, m in rows:
    print(name, 'PASS' if passed else 'FAIL', m)
out = {'best_e1_dir': f'outputs/e2e_improve/{best_name}', 'best_score': best_score, 'runs': [
    {'name': n, 'E1_PASS': p, 'metrics': m} for n, p, m in rows]}
Path('outputs/e2e_improve/e1_best.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
print('Best:', out['best_e1_dir'])
PY

echo "Done. See outputs/e2e_improve/e1_best.json"
