#!/usr/bin/env bash
# F1-4 + F1-6 eval for iter_40000 (run inside Docker segman_train)
set -euo pipefail
cd /workspace/segman
source /root/anaconda3/etc/profile.d/conda.sh

LOG=outputs/e2e_improve/f1_eval_iter40000.log
mkdir -p outputs/e2e_improve segmentation/outputs/m2f_trans10k_pseudo/infer_iter40000

exec > >(tee -a "${LOG}") 2>&1

echo "========== F1-4 infer $(date) =========="
conda activate segman_mmdet
python segmentation/tools/infer_m2f_export_coco.py \
  --config segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py \
  --checkpoint segmentation/outputs/m2f_trans10k_pseudo/iter_40000.pth \
  --ann segmentation/data/trans10k/coco_instances/val.json \
  --out-dir segmentation/outputs/m2f_trans10k_pseudo/infer_iter40000

echo "========== F1-4 match $(date) =========="
python segmentation/tools/eval_instance_match.py \
  --pred-coco segmentation/outputs/m2f_trans10k_pseudo/infer_iter40000/pred_instances.json \
  --gt-coco segmentation/data/trans10k/coco_instances/val.json \
  --iou-match 0.25 \
  --out outputs/e2e_improve/f1_seg_match_iter40000.json

echo "========== F1-6 E2E $(date) =========="
python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --instance-source m2f \
  --m2f-config segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py \
  --m2f-checkpoint segmentation/outputs/m2f_trans10k_pseudo/iter_40000.pth \
  --out-dir outputs/e2e_improve/f1_m2f_e2e \
  --min-area 128 --nms-iou 0.5 --iou-match 0.25

echo "========== F1-6 summarize + gate $(date) =========="
conda activate segman
python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir outputs/e2e_improve/f1_m2f_e2e
python transgrasp/pipelines/run_f1_gate_check.py

echo "========== DONE $(date) =========="
cat outputs/e2e_improve/f1_seg_match_iter40000.json
echo "---"
cat outputs/e2e_improve/f1_plan_summary.json
