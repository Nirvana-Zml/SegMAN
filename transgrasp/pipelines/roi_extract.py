"""Extract instance ROI crops from a semantic label map (Trans10K 11-class)."""
from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as e:
    raise ImportError('opencv-python is required: pip install opencv-python-headless') from e

from transgrasp.pipelines.roi_postprocess import (
    CLASSES,
    FOREGROUND_IDS,
    ExtractConfig,
    InstanceROI,
    mask_iou,
    postprocess_instances,
)

__all__ = [
    'CLASSES', 'FOREGROUND_IDS', 'ExtractConfig', 'InstanceROI',
    'extract_instance_rois', 'mask_iou', 'padded_bbox',
]


def padded_bbox(x0, y0, x1, y1, w, h, pad_ratio: float):
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(w, x1 + px)
    y1 = min(h, y1 + py)
    return x0, y0, x1, y1


def extract_instance_rois(
    rgb: np.ndarray,
    label: np.ndarray,
    bbox_pad: float = 0.15,
    min_area: int = 64,
    extract_cfg: ExtractConfig | None = None,
) -> list[InstanceROI]:
    """Connected components per foreground class; same logic as build_roi_dataset."""
    cfg = extract_cfg or ExtractConfig(bbox_pad=bbox_pad, min_area=min_area)
    if extract_cfg is None:
        cfg.bbox_pad = bbox_pad
        cfg.min_area = min_area

    if label.ndim == 3:
        label = label[..., 0]
    label = label.astype(np.uint8)
    h, w = rgb.shape[:2]
    if label.shape[:2] != (h, w):
        label = np.array(Image.fromarray(label).resize((w, h), Image.NEAREST))

    instances: list[InstanceROI] = []
    counters = {cid: 0 for cid in FOREGROUND_IDS}

    for class_id in FOREGROUND_IDS:
        binary = (label == class_id).astype(np.uint8)
        if binary.sum() == 0:
            continue
        n_comp, comp = cv2.connectedComponents(binary, connectivity=8)
        for comp_id in range(1, n_comp):
            ys, xs = np.where(comp == comp_id)
            area = len(xs)
            min_a = cfg.min_area_per_class.get(CLASSES[class_id], cfg.min_area)
            if area < min_a:
                continue
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, y0, x1, y1 = padded_bbox(x0, y0, x1, y1, w, h, cfg.bbox_pad)
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            class_name = CLASSES[class_id]
            inst_id = counters[class_id]
            counters[class_id] += 1
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[comp == comp_id] = 1
            instances.append(InstanceROI(
                class_id=class_id,
                class_name=class_name,
                instance_id=inst_id,
                bbox=(x0, y0, x1, y1),
                area=area,
                crop_rgb=crop,
                mask=full_mask,
            ))
    return postprocess_instances(instances, cfg, rgb=rgb)
