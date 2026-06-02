#!/usr/bin/env python3
"""Build Copy-Paste patch bank from Trans10K train annotations (Scheme C1)."""
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CLASSES = (
    'background', 'box', 'bottle', 'window', 'eyeglass', 'freezer',
    'jar_kettle', 'door', 'cup', 'wall', 'bowl', 'shelf',
)
# door, shelf, box, freezer, window
DEFAULT_PASTE_CLASSES = (7, 11, 1, 5, 3)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='data/trans10k')
    p.add_argument('--split', type=str, default='train')
    p.add_argument('--out', type=str, default='data/trans10k/copypaste_patch_bank.pkl')
    p.add_argument('--paste-classes', type=str, default='7,11,1,5,3')
    p.add_argument('--min-area', type=int, default=64)
    p.add_argument('--max-area', type=int, default=8192)
    p.add_argument('--max-patches-per-class', type=int, default=400)
    p.add_argument('--pad', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def extract_patches(img: np.ndarray, label: np.ndarray, class_id: int,
                    min_area: int, max_area: int, pad: int) -> list[dict]:
    binary = (label == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return []
    n_comp, comp = cv2.connectedComponents(binary, connectivity=8)
    patches = []
    h, w = label.shape[:2]
    for cid in range(1, n_comp):
        ys, xs = np.where(comp == cid)
        area = len(xs)
        if area < min_area or area > max_area:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x1 + pad)
        y1 = min(h, y1 + pad)
        mask = (comp[y0:y1, x0:x1] == cid).astype(np.uint8)
        crop = img[y0:y1, x0:x1].copy()
        if crop.size == 0 or mask.sum() < min_area:
            continue
        patches.append({
            'class_id': class_id,
            'class_name': CLASSES[class_id],
            'img': crop,
            'mask': mask,
            'area': int(mask.sum()),
        })
    return patches


def main():
    args = parse_args()
    random.seed(args.seed)
    root = Path(args.data_root)
    img_dir = root / 'img_dir' / args.split
    ann_dir = root / 'ann_dir' / args.split
    paste_classes = tuple(int(x) for x in args.paste_classes.split(',') if x.strip())

    bank: dict[int, list] = {c: [] for c in paste_classes}
    stems = sorted({p.stem for p in img_dir.glob('*.jpg')} | {p.stem for p in img_dir.glob('*.png')})

    for stem in stems:
        img_path = img_dir / f'{stem}.jpg'
        if not img_path.is_file():
            img_path = img_dir / f'{stem}.png'
        ann_path = ann_dir / f'{stem}.png'
        if not img_path.is_file() or not ann_path.is_file():
            continue
        img = np.array(Image.open(img_path).convert('RGB'))
        label = np.array(Image.open(ann_path))
        if label.ndim == 3:
            label = label[..., 0]
        for cid in paste_classes:
            if len(bank[cid]) >= args.max_patches_per_class:
                continue
            for patch in extract_patches(
                    img, label.astype(np.uint8), cid,
                    args.min_area, args.max_area, args.pad):
                bank[cid].append(patch)
                if len(bank[cid]) >= args.max_patches_per_class:
                    break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for cid in paste_classes:
        random.shuffle(bank[cid])
        flat.extend(bank[cid][: args.max_patches_per_class])
    with out_path.open('wb') as f:
        pickle.dump(flat, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Wrote {len(flat)} patches -> {out_path}')
    for cid in paste_classes:
        n = sum(1 for p in flat if p['class_id'] == cid)
        print(f'  {CLASSES[cid]}: {n}')


if __name__ == '__main__':
    main()
