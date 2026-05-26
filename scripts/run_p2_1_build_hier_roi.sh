#!/usr/bin/env bash
# P2-1: build hierarchical ROI labels (GT + SegMAN)
set -euo pipefail
cd "$(dirname "$0")/.."

python transgrasp/data/build_hierarchical_roi_labels.py \
  --roi-root data/trans10k_roi_gt \
  --out-root data/trans10k_roi_gt_hier \
  --force

echo "=== GT val stats ==="
python transgrasp/data/stats_roi_dataset.py \
  --root data/trans10k_roi_gt_hier --split val --hierarchical

echo "=== SegMAN val (train split N/A) ==="
python transgrasp/data/build_hierarchical_roi_labels.py \
  --roi-root data/trans10k_roi_segman \
  --out-root data/trans10k_roi_segman_hier \
  --splits val \
  --force

python transgrasp/data/stats_roi_dataset.py \
  --root data/trans10k_roi_segman_hier --split val --hierarchical

echo "P2-1 done."
