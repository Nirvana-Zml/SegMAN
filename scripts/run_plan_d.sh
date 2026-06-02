#!/usr/bin/env bash
# Scheme D: inference-side mask refine (D0–D6)
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate segman 2>/dev/null || true

IMPROVE=outputs/e2e_improve
mkdir -p "${IMPROVE}"

SEG=segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth
if [[ ! -f "${SEG}" ]]; then
  echo "Missing seg checkpoint: ${SEG}" >&2
  exit 1
fi

B1_ARGS=(
  --eval-split val --max-images -1
  --seg-checkpoint "${SEG}"
  --min-area 128
  --nms-iou 0.5
  --max-aspect-ratio 10
  --iou-match 0.25
  --min-area-shelf 32
  --match-algorithm greedy
)

MORPH_ARGS=(
  --refine-morph-close 5
  --refine-morph-classes wall,door,window
)

DILATE_ARGS=(
  --refine-dilate wall:2,door:2,window:1
)

run_eval() {
  local name="$1"
  shift
  echo ""
  echo "========== ${name} =========="
  python transgrasp/pipelines/segment_and_classify.py \
    --out-dir "${IMPROVE}/${name}" \
    "${B1_ARGS[@]}" \
    "$@"
  python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${IMPROVE}/${name}"
}

# D0: B1 baseline (no refine)
run_eval d0_b1_ref

# D1: morph close
run_eval d1_morph "${MORPH_ARGS[@]}"

# D2: morph + dilate (primary combo)
run_eval d2_morph_dilate "${MORPH_ARGS[@]}" "${DILATE_ARGS[@]}"

# D2 grid: lighter dilate
run_eval d2_dilate_wall1_door1 \
  "${MORPH_ARGS[@]}" \
  --refine-dilate wall:1,door:1

# D3: morph + dilate + CRF (optional; skip if pydensecrf missing)
if python -c "import pydensecrf" 2>/dev/null; then
  run_eval d3_crf_on \
    "${MORPH_ARGS[@]}" "${DILATE_ARGS[@]}" \
    --refine-crf --refine-crf-iters 5 \
    --refine-crf-classes wall,door,window
else
  echo "Skip d3_crf_on (pydensecrf not installed)"
fi

# D4: TTA only (slow; baseline postprocess)
run_eval d4_tta_on --seg-tta --seg-tta-scales 0.75,1.0,1.25

# D5: morph + dilate + door-wall split
run_eval d5_split_dw \
  "${MORPH_ARGS[@]}" "${DILATE_ARGS[@]}" \
  --refine-split-door-wall

# D6: gate check + audit on best candidate
python transgrasp/pipelines/run_d_gate_check.py

BEST=$(python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/e2e_improve/d_plan_summary.json')
if p.is_file():
    d = json.loads(p.read_text(encoding='utf-8'))
    b = d.get('best_run') or {}
    print(b.get('name') or 'd2_morph_dilate')
else:
    print('d2_morph_dilate')
PY
)

python transgrasp/pipelines/export_unmatched_instances.py \
  --eval-dir "${IMPROVE}/${BEST}" \
  --out-dir "${IMPROVE}/e2_audit_d_best" \
  --sample-wall 100 --sample-door 50

echo ""
echo "Scheme D done. See outputs/e2e_improve/d_plan_summary.json (best=${BEST})"
