#!/usr/bin/env bash
# P0: m2f_score_thresh / min_area / nms_iou sweep (F1_匹配后分类优化方案.md §4)
set -euo pipefail
cd "$(dirname "$0")/.."

source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
if [[ -f /root/anaconda3/etc/profile.d/conda.sh ]]; then
  source /root/anaconda3/etc/profile.d/conda.sh
fi

CKPT=segmentation/outputs/m2f_trans10k_pseudo/iter_40000.pth
M2F_CFG=segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py
LOG=outputs/e2e_improve/f1_p0_sweep.log
mkdir -p outputs/e2e_improve

exec >> "${LOG}" 2>&1
echo "========== P0 sweep start $(date) =========="

run_one() {
  local id="$1" thresh="$2" min_area="$3" nms="$4" out_dir="$5"
  echo ""
  echo "========== ${id} thresh=${thresh} min_area=${min_area} nms=${nms} $(date) =========="
  conda activate segman_mmdet
  python transgrasp/pipelines/segment_and_classify.py \
    --eval-split val --max-images -1 \
    --instance-source m2f \
    --m2f-config "${M2F_CFG}" \
    --m2f-checkpoint "${CKPT}" \
    --m2f-score-thresh "${thresh}" \
    --out-dir "outputs/e2e_improve/${out_dir}" \
    --min-area "${min_area}" --nms-iou "${nms}" --iou-match 0.25
  conda activate segman
  python transgrasp/pipelines/summarize_e2e_eval.py \
    --eval-dir "outputs/e2e_improve/${out_dir}"
  echo "========== ${id} done $(date) =========="
}

# P0-a: baseline (skip if summary exists)
if [[ ! -f outputs/e2e_improve/f1_m2f_e2e/summary.json ]]; then
  run_one P0-a 0.30 128 0.5 f1_m2f_e2e_thresh0.30
else
  echo "P0-a: reuse existing outputs/e2e_improve/f1_m2f_e2e"
fi

run_one P0-b 0.35 128 0.5 f1_m2f_e2e_thresh0.35
run_one P0-c 0.40 128 0.5 f1_m2f_e2e_thresh0.40
run_one P0-d 0.45 128 0.5 f1_m2f_e2e_thresh0.45
run_one P0-e 0.40 160 0.5 f1_m2f_e2e_thresh0.40_ma160
run_one P0-f 0.35 128 0.6 f1_m2f_e2e_thresh0.35_nms06

conda activate segman
python scripts/collect_f1_p0_results.py

echo "========== P0 sweep DONE $(date) =========="
