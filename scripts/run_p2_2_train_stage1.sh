#!/usr/bin/env bash
# P2-2: train Stage-1 structure vs object router
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/openclip_classifier/p2_stage1_router

python transgrasp/classification/train_hier_stage1.py \
  --config transgrasp/classification/configs/p2_stage1_router.yaml \
  2>&1 | tee outputs/openclip_classifier/p2_stage1_router/train.log

python - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path('outputs/openclip_classifier/p2_stage1_router/eval_val/gate.json').read_text())
print('--- P2-2 gate ---')
print(json.dumps(gate, indent=2))
print('PASS' if gate.get('gate_pass') else 'FAIL')
PY

echo "P2-2 done."
