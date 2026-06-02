#!/usr/bin/env bash
# Scheme F1: Mask2Former full train + seg match + E2E eval
set -euo pipefail
cd "$(dirname "$0")/.."

F1_ENV="${F1_ENV:-segman_mmdet}"
SEG_ENV="${SEG_ENV:-segman}"

source /root/anaconda3/etc/profile.d/conda.sh

IMPROVE=outputs/e2e_improve
M2F_CFG=segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py
M2F_WD=segmentation/outputs/m2f_trans10k_pseudo
M2F_CKPT="${M2F_CKPT:-${M2F_WD}/best_bbox_mAP.pth}"

B1_ARGS=(
  --eval-split val --max-images -1
  --min-area 128
  --nms-iou 0.5
  --max-aspect-ratio 10
  --iou-match 0.25
  --min-area-shelf 32
  --match-algorithm greedy
)

run_eval_segman() {
  conda activate "${SEG_ENV}"
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

echo "========== F1-0 COCO check =========="
conda activate "${SEG_ENV}"
python segmentation/tools/export_trans10k_coco_instances.py \
  --data-root segmentation/data/trans10k \
  --splits train,val \
  --min-area 64 \
  --out-dir segmentation/data/trans10k/coco_instances

run_eval_segman f1_b1_ref --instance-source semantic \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "========== F1-3 Mask2Former train =========="
  conda activate "${F1_ENV}"
  python segmentation/tools/train_m2f_trans10k.py \
    --config "${M2F_CFG}" \
    --work-dir "${M2F_WD}" \
    --batch-size "${BATCH_SIZE:-2}"
fi

echo "========== F1-4 seg-level match =========="
conda activate "${F1_ENV}"
for CK in "${M2F_WD}"/iter_*.pth "${M2F_WD}"/best*.pth; do
  [[ -f "${CK}" ]] || continue
  IT=$(basename "${CK}" .pth)
  OUT="${M2F_WD}/infer_${IT}"
  python segmentation/tools/infer_m2f_export_coco.py \
    --config "${M2F_CFG}" --checkpoint "${CK}" \
    --ann segmentation/data/trans10k/coco_instances/val.json \
    --out-dir "${OUT}"
  python segmentation/tools/eval_instance_match.py \
    --pred-coco "${OUT}/pred_instances.json" \
    --gt-coco segmentation/data/trans10k/coco_instances/val.json \
    --iou-match 0.25 \
    --out "${IMPROVE}/f1_seg_match_${IT}.json"
done

if [[ -f "${M2F_CKPT}" ]]; then
  echo "========== F1-6 m2f E2E =========="
  conda activate "${F1_ENV}"
  python transgrasp/pipelines/segment_and_classify.py \
    --out-dir "${IMPROVE}/f1_m2f_e2e" \
    "${B1_ARGS[@]}" \
    --instance-source m2f \
    --m2f-config "${M2F_CFG}" \
    --m2f-checkpoint "${M2F_CKPT}"
  python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${IMPROVE}/f1_m2f_e2e"
else
  echo "Skip f1_m2f_e2e: no checkpoint at ${M2F_CKPT}"
fi

conda activate "${SEG_ENV}"
python transgrasp/pipelines/run_f1_gate_check.py

BEST=$(python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/e2e_improve/f1_plan_summary.json')
if p.is_file():
    b = json.loads(p.read_text()).get('best_run') or {}
    print(b.get('name') or 'f1_b1_ref')
else:
    print('f1_b1_ref')
PY
)

python transgrasp/pipelines/export_unmatched_instances.py \
  --eval-dir "${IMPROVE}/${BEST}" \
  --out-dir "${IMPROVE}/e2_audit_f1_best" \
  --sample-wall 100 --sample-door 50 2>/dev/null || true

echo ""
echo "Scheme F1 done. See ${IMPROVE}/f1_plan_summary.json (best=${BEST})"
