"""ROI instance types and post-processing (E1)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLASSES = (
    'background', 'box', 'bottle', 'window', 'eyeglass', 'freezer',
    'jar_kettle', 'door', 'cup', 'wall', 'bowl', 'shelf',
)
FOREGROUND_IDS = tuple(range(1, len(CLASSES)))


@dataclass
class InstanceROI:
    class_id: int
    class_name: str
    instance_id: int
    bbox: tuple[int, int, int, int]
    area: int
    crop_rgb: np.ndarray
    mask: np.ndarray

    def to_dict(self) -> dict:
        x0, y0, x1, y1 = self.bbox
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'instance_id': self.instance_id,
            'bbox': [x0, y0, x1, y1],
            'area': self.area,
        }


@dataclass
class ExtractConfig:
    bbox_pad: float = 0.15
    min_area: int = 64
    min_area_per_class: dict[str, int] = field(default_factory=dict)
    max_aspect_ratio: float = 10.0
    nms_iou: float = 0.5
    enable_nms: bool = False
    merge_cc_iou: float = 0.0
    merge_cc_dist: int = 8
    merge_cc_classes: tuple[str, ...] = ()


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(ax1 - ax0, 1) * max(ay1 - ay0, 1)
    area_b = max(bx1 - bx0, 1) * max(by1 - by0, 1)
    return inter / (area_a + area_b - inter)


def filter_instances(instances: list[InstanceROI], cfg: ExtractConfig) -> list[InstanceROI]:
    out: list[InstanceROI] = []
    for inst in instances:
        min_a = cfg.min_area_per_class.get(inst.class_name, cfg.min_area)
        if inst.area < min_a:
            continue
        x0, y0, x1, y1 = inst.bbox
        bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
        ar = max(bw / bh, bh / bw)
        if cfg.max_aspect_ratio > 0 and ar > cfg.max_aspect_ratio:
            continue
        out.append(inst)
    return out


def nms_instances(instances: list[InstanceROI], iou_thresh: float) -> list[InstanceROI]:
    if iou_thresh <= 0 or len(instances) <= 1:
        return instances
    by_class: dict[int, list[InstanceROI]] = {}
    for inst in instances:
        by_class.setdefault(inst.class_id, []).append(inst)

    kept: list[InstanceROI] = []
    for class_id in sorted(by_class):
        group = sorted(by_class[class_id], key=lambda x: x.area, reverse=True)
        suppress = [False] * len(group)
        for i in range(len(group)):
            if suppress[i]:
                continue
            kept.append(group[i])
            for j in range(i + 1, len(group)):
                if suppress[j]:
                    continue
                if bbox_iou(group[i].bbox, group[j].bbox) >= iou_thresh:
                    suppress[j] = True
    return kept


def bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Min edge distance between two bboxes (0 if overlapping)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    return int(dx + dy)


def _merge_pair(a: InstanceROI, b: InstanceROI, rgb: np.ndarray | None) -> InstanceROI:
    mask = np.logical_or(a.mask > 0, b.mask > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    area = int(len(xs))
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if rgb is not None:
        crop = rgb[y0:y1, x0:x1].copy()
    else:
        crop = a.crop_rgb if a.area >= b.area else b.crop_rgb
    return InstanceROI(
        class_id=a.class_id,
        class_name=a.class_name,
        instance_id=a.instance_id,
        bbox=(x0, y0, x1, y1),
        area=area,
        crop_rgb=crop,
        mask=mask,
    )


def merge_nearby_cc(
    instances: list[InstanceROI],
    merge_iou: float,
    merge_dist: int,
    class_names: tuple[str, ...],
    rgb: np.ndarray | None = None,
) -> list[InstanceROI]:
    """Merge same-class CCs with bbox IoU >= merge_iou or edge gap <= merge_dist."""
    if merge_iou <= 0 or not instances:
        return instances
    allowed = set(class_names) if class_names else None

    by_class: dict[int, list[InstanceROI]] = {}
    passthrough: list[InstanceROI] = []
    for inst in instances:
        if allowed is not None and inst.class_name not in allowed:
            passthrough.append(inst)
            continue
        by_class.setdefault(inst.class_id, []).append(inst)

    merged: list[InstanceROI] = list(passthrough)
    for group in by_class.values():
        active = list(group)
        progress = True
        while progress and len(active) > 1:
            progress = False
            used = [False] * len(active)
            next_round: list[InstanceROI] = []
            for i in range(len(active)):
                if used[i]:
                    continue
                cur = active[i]
                for j in range(i + 1, len(active)):
                    if used[j]:
                        continue
                    other = active[j]
                    if (bbox_iou(cur.bbox, other.bbox) >= merge_iou or
                            (merge_dist > 0 and bbox_gap(cur.bbox, other.bbox) <= merge_dist)):
                        cur = _merge_pair(cur, other, rgb)
                        used[j] = True
                        progress = True
                next_round.append(cur)
                used[i] = True
            active = next_round
        merged.extend(active)
    return merged


def postprocess_instances(
    instances: list[InstanceROI],
    cfg: ExtractConfig,
    rgb: np.ndarray | None = None,
) -> list[InstanceROI]:
    inst = filter_instances(instances, cfg)
    if cfg.merge_cc_iou > 0:
        inst = merge_nearby_cc(
            inst, cfg.merge_cc_iou, cfg.merge_cc_dist,
            cfg.merge_cc_classes, rgb=rgb)
    if cfg.enable_nms and cfg.nms_iou > 0:
        inst = nms_instances(inst, cfg.nms_iou)
    return inst


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter) / float(union)
