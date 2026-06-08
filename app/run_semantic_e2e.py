#!/usr/bin/env python3
"""TransGrasp deliver V1.0 — mode A: SegMAN semantic seg + OpenCLIP classify."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Inject mode-A defaults before segment_and_classify parses argv.
_DEFAULTS = [
    '--instance-source', 'semantic',
    '--seg-config',
    'segmentation/local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py',
    '--seg-checkpoint',
    'segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth',
    '--cls-checkpoint', 'outputs/openclip_classifier/deliver_classifier_best.pth',
    '--class-thresholds',
    'transgrasp/classification/configs/reject_thresholds_p3.json',
    '--min-area', '128',
    '--nms-iou', '0.5',
    '--bbox-pad', '0.15',
]

argv = sys.argv[1:]
if not any(a.startswith('--instance-source') for a in argv):
    sys.argv = [sys.argv[0], *_DEFAULTS, *argv]

runpy.run_path(
    str(ROOT / 'transgrasp/pipelines/segment_and_classify.py'),
    run_name='__main__',
)
