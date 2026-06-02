#!/usr/bin/env bash
# Scheme E: pseudo COCO export + Mask R-CNN (E1-lite) + E2E eval
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate segman 2>/dev/null || true

IMPROVE=outputs/e2e_improve
mkdir -p "${IMPROVE}"

B1_ARGS=(
  --eval-split val --max-images -1
  --min-area 128
  --nms-iou 0.5
  --max-aspect-ratio 10
  --iou-match 0.25
  --min-area-shelf 32
  --match-algorithm greedy
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

echo "========== E0 COCO export =========="
python segmentation/tools/export_trans10k_coco_instances.py \
  --data-root segmentation/data/trans10k \
  --splits train,val \
  --min-area 64 \
  --out-dir segmentation/data/trans10k/coco_instances

python segmentation/tools/browse_coco_instances.py \
  --ann segmentation/data/trans10k/coco_instances/val.json \
  --img-dir segmentation/data/trans10k/img_dir/val \
  --max-images 10 \
  --out-dir "${IMPROVE}/e0_coco_browse"

echo "========== E0 B1 baseline =========="
run_eval e0_b1_ref --instance-source semantic \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth

echo "========== E2 GT oracle upper bound =========="
run_eval e2_gt_oracle --instance-source gt_oracle

MRCNN_CKPT=segmentation/outputs/maskrcnn_trans10k_pseudo/best.pth
if [[ ! -f "${MRCNN_CKPT}" ]]; then
  echo "========== E1 Mask R-CNN train (E1-lite) =========="
  python segmentation/tools/train_maskrcnn_trans10k.py \
    --epochs "${EPOCHS:-15}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --out-dir segmentation/outputs/maskrcnn_trans10k_pseudo
fi

if [[ -f "${MRCNN_CKPT}" ]]; then
  echo "========== E4 Mask R-CNN E2E =========="
  run_eval e4_maskrcnn_e2e \
    --instance-source maskrcnn \
    --maskrcnn-checkpoint "${MRCNN_CKPT}"
else
  echo "Skip e4_maskrcnn_e2e: no checkpoint"
fi

python transgrasp/pipelines/run_e_gate_check.py

BEST=$(python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/e2e_improve/e_plan_summary.json')
if p.is_file():
    b = json.loads(p.read_text()).get('best_run') or {}
    print(b.get('name') or 'e2_gt_oracle')
else:
    print('e2_gt_oracle')
PY
)

python transgrasp/pipelines/export_unmatched_instances.py \
  --eval-dir "${IMPROVE}/${BEST}" \
  --out-dir "${IMPROVE}/e2_audit_e_best" \
  --sample-wall 100 --sample-door 50 2>/dev/null || true

echo ""
echo "Scheme E done. See outputs/e2e_improve/e_plan_summary.json (best=${BEST})"
