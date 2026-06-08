#!/usr/bin/env python3
"""TransGrasp deliver V1.0 — mode B: M2F instance + OpenCLIP classify (grasp recommended)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DEFAULTS = [
    '--instance-source', 'm2f',
    '--m2f-config',
    'segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py',
    '--m2f-checkpoint',
    'segmentation/outputs/m2f_trans10k_pseudo/iter_40000.pth',
    '--m2f-score-thresh', '0.30',
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
