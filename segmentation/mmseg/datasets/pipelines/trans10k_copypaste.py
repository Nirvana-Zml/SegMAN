# Copyright (c) OpenMMLab. All rights reserved.
"""Trans10K Copy-Paste augmentation for weak-class recall (Scheme C0)."""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import cv2
import mmcv
import numpy as np

from ..builder import PIPELINES

_BANK_CACHE: dict[str, list] = {}


def _load_bank(path: str) -> list:
    if path not in _BANK_CACHE:
        with open(path, 'rb') as f:
            _BANK_CACHE[path] = pickle.load(f)
    return _BANK_CACHE[path]


@PIPELINES.register_module()
class Trans10KCopyPaste(object):
    """Paste weak-class instance patches onto train images."""

    def __init__(
        self,
        patch_bank: str = 'data/trans10k/copypaste_patch_bank.pkl',
        paste_prob: float = 0.5,
        max_paste: int = 2,
        paste_classes=None,
        scale_range=(0.8, 1.2),
        max_overlap: float = 0.3,
        max_retry: int = 10,
    ):
        self.patch_bank = patch_bank
        self.paste_prob = paste_prob
        self.max_paste = max_paste
        self.paste_classes = paste_classes
        self.scale_range = scale_range
        self.max_overlap = max_overlap
        self.max_retry = max_retry
        self._patches: list | None = None
        self._by_class: dict[int, list] | None = None

    def _ensure_bank(self):
        if self._patches is not None:
            return
        bank_path = self.patch_bank
        if not Path(bank_path).is_file():
            # resolve relative to cwd (segmentation/)
            alt = Path(__file__).resolve().parents[3] / bank_path
            bank_path = str(alt if alt.is_file() else bank_path)
        self._patches = _load_bank(bank_path)
        self._by_class = {}
        for p in self._patches:
            self._by_class.setdefault(p['class_id'], []).append(p)
        if self.paste_classes is not None:
            allowed = set(int(c) for c in self.paste_classes)
            self._patches = [p for p in self._patches if p['class_id'] in allowed]
            self._by_class = {k: v for k, v in self._by_class.items() if k in allowed}

    def _resize_patch(self, img: np.ndarray, mask: np.ndarray, scale: float):
        h, w = img.shape[:2]
        nh = max(4, int(h * scale))
        nw = max(4, int(w * scale))
        img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        mask_r = (mask_r > 0).astype(np.uint8)
        return img_r, mask_r

    def _overlap_ratio(self, seg: np.ndarray, mask: np.ndarray, x0, y0) -> float:
        h, w = mask.shape[:2]
        H, W = seg.shape[:2]
        if x0 + w > W or y0 + h > H or x0 < 0 or y0 < 0:
            return 1.0
        region = seg[y0:y0 + h, x0:x0 + w]
        fg = region > 0
        if fg.sum() == 0:
            return 0.0
        inter = np.logical_and(fg, mask > 0).sum()
        return float(inter) / float(fg.sum())

    def _paste_once(self, img: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._patches and self._by_class
        cid = random.choice(list(self._by_class.keys()))
        patch = random.choice(self._by_class[cid])
        scale = random.uniform(*self.scale_range)
        p_img, p_mask = self._resize_patch(patch['img'], patch['mask'], scale)
        ph, pw = p_img.shape[:2]
        H, W = img.shape[:2]
        if ph >= H or pw >= W:
            return img, seg

        for _ in range(self.max_retry):
            x0 = random.randint(0, W - pw)
            y0 = random.randint(0, H - ph)
            if self._overlap_ratio(seg, p_mask, x0, y0) > self.max_overlap:
                continue
            roi = img[y0:y0 + ph, x0:x0 + pw]
            m = p_mask.astype(bool)
            roi[m] = p_img[m]
            seg[y0:y0 + ph, x0:x0 + pw][m] = cid
            return img, seg
        return img, seg

    def __call__(self, results: dict) -> dict:
        if random.random() > self.paste_prob:
            return results
        self._ensure_bank()
        if not self._patches:
            return results

        img = results['img']
        seg = results['gt_semantic_seg']
        n = random.randint(1, self.max_paste)
        for _ in range(n):
            img, seg = self._paste_once(img, seg)
        results['img'] = img
        results['gt_semantic_seg'] = seg
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(paste_prob={self.paste_prob}, '
                f'max_paste={self.max_paste}, bank={self.patch_bank})')
