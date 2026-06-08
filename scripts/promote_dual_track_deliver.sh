#!/usr/bin/env bash
# Verify dual-track deliver artifacts and refresh manifest timestamp.
set -euo pipefail
cd "$(dirname "$0")/.."

check() {
  if [[ ! -f "$1" ]]; then
    echo "Error: missing $1" >&2
    exit 1
  fi
}

check segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth
check segmentation/outputs/m2f_trans10k_pseudo/iter_40000.pth
check outputs/openclip_classifier/deliver_classifier_best.pth
check transgrasp/classification/configs/reject_thresholds_p3.json
check app/run_semantic_e2e.py
check app/run_grasp_e2e.py

python - <<'PY'
import json
from datetime import date
from pathlib import Path

p = Path('outputs/e2e_improve/deliver_dual_track_manifest.json')
m = json.loads(p.read_text(encoding='utf-8'))
m['promoted_at'] = str(date.today())
m['status'] = 'current_dual_track_deliver'
p.write_text(json.dumps(m, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('OK:', p)
PY

echo "Dual-track deliver V1.0.0 verified."
echo "  Doc: docs/交付/SegMAN_OpenCLIP_E2E_交付路线.md"
echo "  Mode A: app/run_semantic_e2e.py"
echo "  Mode B: app/run_grasp_e2e.py"
