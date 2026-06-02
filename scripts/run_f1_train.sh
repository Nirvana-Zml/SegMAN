#!/usr/bin/env bash
# F1-3: start Mask2Former 40k iter training (background-friendly)
set -euo pipefail
cd "$(dirname "$0")/.."

source /root/anaconda3/etc/profile.d/conda.sh
conda activate segman_mmdet

LOG=outputs/e2e_improve/f1_m2f_train.log
mkdir -p outputs/e2e_improve segmentation/outputs/m2f_trans10k_pseudo

echo "F1-3 Mask2Former 40k iter started $(date)" | tee -a "${LOG}"
python segmentation/tools/train_m2f_trans10k.py \
  --config segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py \
  --work-dir segmentation/outputs/m2f_trans10k_pseudo \
  --batch-size "${BATCH_SIZE:-2}" \
  2>&1 | tee -a "${LOG}"
echo "F1-3 done $(date)" | tee -a "${LOG}"
