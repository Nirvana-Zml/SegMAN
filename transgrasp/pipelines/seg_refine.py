"""Semantic label-map refinement for scheme D (morph / dilate / CRF / door-wall split)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError('opencv-python is required: pip install opencv-python-headless') from e

from transgrasp.pipelines.roi_postprocess import CLASSES

NAME_TO_ID = {name: i for i, name in enumerate(CLASSES)}


@dataclass
class RefineConfig:
    morph_close: int = 0
    morph_classes: tuple[str, ...] = ()
    dilate: dict[str, int] = field(default_factory=dict)
    erode: dict[str, int] = field(default_factory=dict)
    crf: bool = False
    crf_iters: int = 5
    crf_classes: tuple[str, ...] = ()
    split_door_wall: bool = False


def parse_class_px_spec(spec: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if not spec.strip():
        return out
    for part in spec.split(','):
        part = part.strip()
        if not part or ':' not in part:
            continue
        name, val = part.split(':', 1)
        name = name.strip()
        if name not in NAME_TO_ID:
            raise ValueError(f'Unknown class {name!r} in refine spec')
        out[name] = int(val.strip())
    return out


def parse_class_list(spec: str) -> tuple[str, ...]:
    if not spec.strip():
        return ()
    names = tuple(c.strip() for c in spec.split(',') if c.strip())
    for name in names:
        if name not in NAME_TO_ID:
            raise ValueError(f'Unknown class {name!r}')
    return names


def _kernel(size: int):
    k = max(int(size), 1)
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def morph_close_label_map(
    label: np.ndarray,
    kernel_size: int,
    class_names: tuple[str, ...],
) -> np.ndarray:
    if kernel_size <= 0 or not class_names:
        return label
    out = label.copy()
    kernel = _kernel(kernel_size)
    for name in class_names:
        cid = NAME_TO_ID[name]
        binary = (out == cid).astype(np.uint8)
        if binary.sum() == 0:
            continue
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        fill = (closed > 0) & (out == 0)
        out[fill] = cid
    return out


def dilate_erode_label_map(
    label: np.ndarray,
    dilate: dict[str, int] | None = None,
    erode: dict[str, int] | None = None,
) -> np.ndarray:
    out = label.copy()
    dilate = dilate or {}
    erode = erode or {}

    for name, px in erode.items():
        if px <= 0:
            continue
        cid = NAME_TO_ID[name]
        binary = (out == cid).astype(np.uint8)
        if binary.sum() == 0:
            continue
        shrunk = cv2.erode(binary, _kernel(2 * px + 1))
        out[(binary > 0) & (shrunk == 0)] = 0

    for name, px in dilate.items():
        if px <= 0:
            continue
        cid = NAME_TO_ID[name]
        binary = (out == cid).astype(np.uint8)
        if binary.sum() == 0:
            continue
        expanded = cv2.dilate(binary, _kernel(2 * px + 1))
        grow = (expanded > 0) & (out == 0)
        out[grow] = cid
    return out


def split_door_wall_label_map(label: np.ndarray) -> np.ndarray:
    """Peel door pixels adjacent to wall (adhesion heuristic)."""
    door_id = NAME_TO_ID['door']
    wall_id = NAME_TO_ID['wall']
    out = label.copy()
    door_mask = (out == door_id).astype(np.uint8)
    if door_mask.sum() == 0:
        return out
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    door_eroded = cv2.erode(door_mask, kernel)
    peel = (door_mask > 0) & (door_eroded == 0)
    wall_near = cv2.dilate((out == wall_id).astype(np.uint8), kernel) > 0
    reassign = peel & wall_near
    out[reassign] = wall_id
    return out


def dense_crf_refine(
    rgb: np.ndarray,
    label: np.ndarray,
    n_iters: int = 5,
    restrict_classes: tuple[str, ...] = (),
) -> np.ndarray:
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels
    except ImportError as e:
        raise ImportError('pydensecrf required: pip install pydensecrf') from e

    h, w = label.shape[:2]
    n_classes = len(CLASSES)
    unary = unary_from_labels(label.astype(np.int32), n_classes, gt_prob=0.7)
    d = dcrf.DenseCRF2D(w, h, n_classes)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(
        sxy=80, srgb=13, rgbim=np.ascontiguousarray(rgb, dtype=np.uint8), compat=10)
    refined = d.inference(n_iters)
    crf_label = np.argmax(refined, axis=0).reshape(h, w).astype(np.uint8)

    if restrict_classes:
        allowed = {NAME_TO_ID[c] for c in restrict_classes}
        out = label.copy()
        change = np.zeros((h, w), dtype=bool)
        for cid in allowed:
            change |= (label == cid)
        out[change] = crf_label[change]
        return out
    return crf_label


def apply_seg_refine(rgb: np.ndarray, label: np.ndarray, cfg: RefineConfig) -> np.ndarray:
    if not any([
        cfg.morph_close > 0 and cfg.morph_classes,
        cfg.dilate,
        cfg.erode,
        cfg.crf,
        cfg.split_door_wall,
    ]):
        return label

    out = label.astype(np.uint8).copy()
    if cfg.morph_close > 0 and cfg.morph_classes:
        out = morph_close_label_map(out, cfg.morph_close, cfg.morph_classes)
    if cfg.dilate or cfg.erode:
        out = dilate_erode_label_map(out, cfg.dilate, cfg.erode)
    if cfg.split_door_wall:
        out = split_door_wall_label_map(out)
    if cfg.crf:
        out = dense_crf_refine(
            rgb, out, n_iters=cfg.crf_iters, restrict_classes=cfg.crf_classes)
    return out


def build_refine_config(args) -> RefineConfig:
    return RefineConfig(
        morph_close=args.refine_morph_close,
        morph_classes=parse_class_list(args.refine_morph_classes),
        dilate=parse_class_px_spec(args.refine_dilate),
        erode=parse_class_px_spec(args.refine_erode),
        crf=args.refine_crf,
        crf_iters=args.refine_crf_iters,
        crf_classes=parse_class_list(args.refine_crf_classes),
        split_door_wall=args.refine_split_door_wall,
    )
