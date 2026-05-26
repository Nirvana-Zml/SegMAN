#!/bin/bash
# P0-1-3 ~ P0-3 (run inside Docker: segman_train, conda env segman)
# Usage:
#   docker exec -it segman_train bash
#   cd /workspace/segman && bash scripts/run_p0_remaining.sh
set -euo pipefail

ROOT=/workspace/segman
SEG_CKPT_FILE="${ROOT}/segmentation/outputs/trans10k_lass_mmscope_balanced_v2_p0weak/P0_SEG_CKPT.txt"
P0_DIR="${ROOT}/segmentation/outputs/trans10k_lass_mmscope_balanced_v2_p0weak"
CFG=local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2_p0weak.py

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segman
cd "${ROOT}/segmentation"

echo "=== P0-1-3: eval checkpoints ==="
for ckpt in iter_2000 iter_4000 best_mIoU_iter_2000; do
  if [[ -f "${P0_DIR}/${ckpt}.pth" ]]; then
    python tools/test.py "${CFG}" \
      --checkpoint "${P0_DIR}/${ckpt}.pth" \
      --eval mIoU \
      --work-dir "${P0_DIR}/eval_${ckpt}"
  fi
done

python scripts/p0_select_seg_ckpt.py \
  --work-dir "${P0_DIR}" \
  --write "${SEG_CKPT_FILE}"
P0_SEG_CKPT=$(tr -d '\r\n' < "${SEG_CKPT_FILE}")
echo "Selected: ${P0_SEG_CKPT}"

cd "${ROOT}"
echo "=== P0-2-1: export pred masks ==="
python transgrasp/data/export_sem_seg_preds.py \
  --config segmentation/${CFG} \
  --checkpoint "segmentation/${P0_DIR}/${P0_SEG_CKPT}" \
  --data-root segmentation/data/trans10k \
  --split val \
  --out-dir "segmentation/${P0_DIR}/pred_sem_seg_val"

echo "=== P0-2-2: build SegMAN-ROI ==="
python transgrasp/data/build_roi_dataset.py \
  --data-root segmentation/data/trans10k \
  --split val \
  --mask-source segman \
  --pred-dir "segmentation/${P0_DIR}/pred_sem_seg_val" \
  --out-root data/trans10k_roi_segman_p0weak/val \
  --bbox-pad 0.15 \
  --min-area 64

python transgrasp/data/stats_roi_dataset.py \
  --root data/trans10k_roi_segman_p0weak --split val
python transgrasp/data/stats_roi_dataset.py \
  --root data/trans10k_roi_segman --split val

echo "=== P0-3: classify on new ROI ==="
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint outputs/openclip_classifier/deliver_classifier_best.pth \
  --roi-root data/trans10k_roi_segman_p0weak \
  --split val \
  --report-dir outputs/openclip_classifier/p0weak_eval_segman_roi

echo "Done. Compare outputs/openclip_classifier/p0weak_eval_segman_roi/summary.json vs deliver_t2_best."
