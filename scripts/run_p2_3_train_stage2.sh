#!/usr/bin/env bash
# P2-2b: retry Stage-1 with balanced CE; P2-3: structure specialist
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== P2-2b Stage-1 (balanced CE) ==="
mkdir -p outputs/openclip_classifier/p2_stage1_router_balanced
python transgrasp/classification/train_hier_stage1.py \
  --config transgrasp/classification/configs/p2_stage1_router.yaml \
  --work-dir outputs/openclip_classifier/p2_stage1_router_balanced \
  --balance-stage1 \
  2>&1 | tee outputs/openclip_classifier/p2_stage1_router_balanced/train.log

python - <<'PY'
import json
from pathlib import Path
for name in ('p2_stage1_router_balanced', 'p2_stage1_router'):
    p = Path(f'outputs/openclip_classifier/{name}/eval_val/gate.json')
    if p.is_file():
        g = json.loads(p.read_text())
        print(name, 'gate_pass=', g.get('gate_pass'), g)
PY

echo "=== P2-3 Stage-2 structure head ==="
mkdir -p outputs/openclip_classifier/p2_stage2_structure
python transgrasp/classification/train_hier_stage2_structure.py \
  --config transgrasp/classification/configs/p2_stage2_structure.yaml \
  2>&1 | tee outputs/openclip_classifier/p2_stage2_structure/train.log

echo "P2-2b/3 done."
