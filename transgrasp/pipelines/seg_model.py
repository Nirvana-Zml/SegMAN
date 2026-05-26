"""SegMAN (MMSeg) wrapper for single-image semantic segmentation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEG_ROOT = PROJECT_ROOT / 'segmentation'

DEFAULT_SEG_CONFIG = (
    'segmentation/local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py'
)
DEFAULT_SEG_CHECKPOINT = (
    'segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth'
)


def _ensure_mmseg_path():
    seg = str(SEG_ROOT.resolve())
    if seg not in sys.path:
        sys.path.insert(0, seg)


def resolve_seg_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


class SegMANSegmentor:
    def __init__(self, config: str, checkpoint: str, device: str = 'cuda:0'):
        _ensure_mmseg_path()
        from mmseg.apis import init_segmentor

        cfg = resolve_seg_path(config)
        ckpt = resolve_seg_path(checkpoint)
        if not cfg.is_file():
            raise FileNotFoundError(f'Seg config not found: {cfg}')
        if not ckpt.is_file():
            raise FileNotFoundError(f'Seg checkpoint not found: {ckpt}')

        self.device = device
        self.config_path = cfg
        self.checkpoint_path = ckpt
        old_cwd = os.getcwd()
        try:
            os.chdir(SEG_ROOT)
            self.model = init_segmentor(str(cfg), str(ckpt), device=device)
        finally:
            os.chdir(old_cwd)

    def predict_label_map(self, image_path: str | Path | np.ndarray) -> np.ndarray:
        _ensure_mmseg_path()
        from mmseg.apis import inference_segmentor
        import mmcv

        if isinstance(image_path, np.ndarray):
            img = image_path
        else:
            img = mmcv.imread(str(image_path))

        old_cwd = os.getcwd()
        try:
            os.chdir(SEG_ROOT)
            result = inference_segmentor(self.model, img)
        finally:
            os.chdir(old_cwd)

        sem = result[0]
        if hasattr(sem, 'cpu'):
            sem = sem.cpu().numpy()
        return np.asarray(sem, dtype=np.uint8)
