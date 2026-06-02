#!/usr/bin/env bash
# F1: install mmdet 3.x in isolated conda env (keeps segman/mmseg 0.30 + mmcv 1.7 intact)
set -euo pipefail
cd "$(dirname "$0")/.."

source /root/anaconda3/etc/profile.d/conda.sh

ENV_NAME="${F1_ENV:-segman_mmdet}"

if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "Creating conda env ${ENV_NAME}..."
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"

pip install -U pip wheel 'setuptools<81'

# Pin torch 2.1 for mmcv/mmdet prebuilt wheels (cu118)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install 'numpy<2'

pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
pip install "mmdet>=3.0.0"

pip install pycocotools opencv-python-headless open-clip-torch timm scipy

# mmdet configs for _base_ inheritance
MMDET_CFG=segmentation/mmdet_configs
if [[ ! -d "${MMDET_CFG}/mask2former" ]]; then
  echo "Fetching mmdet mask2former configs..."
  mkdir -p "${MMDET_CFG}"
  BASE=https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs
  for f in \
    mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py \
    mask2former/mask2former_r50_8xb2-lsj-50e_coco.py \
    _base_/default_runtime.py \
    _base_/datasets/coco_panoptic.py
  do
    mkdir -p "${MMDET_CFG}/$(dirname "$f")"
    curl -fsSL "${BASE}/${f}" -o "${MMDET_CFG}/${f}"
  done
fi

python -c "import mmdet; import mmengine; import mmcv; print('OK mmdet', mmdet.__version__, 'mmcv', mmcv.__version__)"
echo "F1 env ${ENV_NAME} ready."
