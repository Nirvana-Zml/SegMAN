#!/usr/bin/env python3
"""Export Trans10K semantic GT to COCO instance JSON (scheme E0).

Uses the same CC rules as E2E GT matching (min_area=64, 8-connectivity per class).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SEG_ROOT = Path(__file__).resolve().parents[1]
if str(SEG_ROOT) not in sys.path:
    sys.path.insert(0, str(SEG_ROOT.parent))

from transgrasp.pipelines.roi_postprocess import CLASSES, FOREGROUND_IDS

# COCO category_id = class index (1..11); skip background
COCO_CATEGORIES = [
    {'id': cid, 'name': CLASSES[cid], 'supercategory': 'transparent'}
    for cid in FOREGROUND_IDS
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='segmentation/data/trans10k')
    p.add_argument('--splits', type=str, default='train,val')
    p.add_argument('--min-area', type=int, default=64)
    p.add_argument('--out-dir', type=str, default='segmentation/data/trans10k/coco_instances')
    return p.parse_args()


def resolve_path(p: str | Path, root: Path) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def encode_rle(binary: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils
    binary = np.asfortranarray(binary.astype(np.uint8))
    rle = mask_utils.encode(binary)
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def export_split(data_root: Path, split: str, min_area: int) -> dict:
    img_dir = data_root / 'img_dir' / split
    ann_dir = data_root / 'ann_dir' / split
    images = []
    annotations = []
    ann_id = 1
    per_class = {CLASSES[c]: 0 for c in FOREGROUND_IDS}

    stems = sorted({p.stem for p in img_dir.glob('*.jpg')} | {p.stem for p in img_dir.glob('*.png')})
    for img_id, stem in enumerate(stems, start=1):
        img_path = img_dir / f'{stem}.jpg'
        if not img_path.is_file():
            img_path = img_dir / f'{stem}.png'
        ann_path = ann_dir / f'{stem}.png'
        if not img_path.is_file() or not ann_path.is_file():
            continue

        label = np.array(Image.open(ann_path))
        if label.ndim == 3:
            label = label[..., 0]
        h, w = label.shape[:2]

        images.append({
            'id': img_id,
            'file_name': img_path.name,
            'width': w,
            'height': h,
        })

        for class_id in FOREGROUND_IDS:
            binary = (label == class_id).astype(np.uint8)
            if binary.sum() == 0:
                continue
            n_comp, comp = cv2.connectedComponents(binary, connectivity=8)
            for comp_id in range(1, n_comp):
                ys, xs = np.where(comp == comp_id)
                area = int(len(xs))
                if area < min_area:
                    continue
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                inst_mask = (comp == comp_id).astype(np.uint8)
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': class_id,
                    'bbox': [x0, y0, x1 - x0, y1 - y0],
                    'area': area,
                    'iscrowd': 0,
                    'segmentation': encode_rle(inst_mask),
                })
                per_class[CLASSES[class_id]] += 1
                ann_id += 1

    return {
        'info': {
            'description': f'Trans10K pseudo-instances ({split})',
            'version': '1.0',
            'date_created': datetime.utcnow().isoformat(),
            'min_area': min_area,
        },
        'licenses': [],
        'categories': COCO_CATEGORIES,
        'images': images,
        'annotations': annotations,
        'stats': {
            'num_images': len(images),
            'num_instances': len(annotations),
            'per_class': per_class,
        },
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    data_root = resolve_path(args.data_root, root)
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits.split(','):
        split = split.strip()
        if not split:
            continue
        coco = export_split(data_root, split, args.min_area)
        out_path = out_dir / f'{split}.json'
        out_path.write_text(json.dumps(coco, indent=2) + '\n', encoding='utf-8')
        stats = coco['stats']
        print(f'{split}: images={stats["num_images"]} instances={stats["num_instances"]}')
        print(f'  per_class: {stats["per_class"]}')
        print(f'  -> {out_path}')


if __name__ == '__main__':
    main()
