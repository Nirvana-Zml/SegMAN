#!/usr/bin/env python3
"""Browse COCO pseudo-instance annotations (scheme E0 QA)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0),
    (255, 0, 128),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ann', type=str, required=True)
    p.add_argument('--img-dir', type=str, required=True)
    p.add_argument('--max-images', type=int, default=20)
    p.add_argument('--out-dir', type=str, default='outputs/e2e_improve/e0_coco_browse')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def decode_rle(seg, h, w):
    from pycocotools import mask as mask_utils
    if isinstance(seg, dict):
        return mask_utils.decode(seg)
    return None


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    ann_path = Path(args.ann)
    if not ann_path.is_absolute():
        ann_path = root / ann_path
    img_dir = Path(args.img_dir)
    if not img_dir.is_absolute():
        img_dir = root / img_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(ann_path.read_text(encoding='utf-8'))
    cat_names = {c['id']: c['name'] for c in coco['categories']}
    anns_by_img: dict[int, list] = {}
    for ann in coco['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    images = coco['images']
    random.seed(args.seed)
    sample = random.sample(images, min(args.max_images, len(images)))

    for img_info in sample:
        img_path = img_dir / img_info['file_name']
        if not img_path.is_file():
            stem = Path(img_info['file_name']).stem
            for ext in ('.jpg', '.png'):
                alt = img_dir / f'{stem}{ext}'
                if alt.is_file():
                    img_path = alt
                    break
        rgb = np.array(Image.open(img_path).convert('RGB'))
        h, w = rgb.shape[:2]
        vis = rgb.copy()
        for ann in anns_by_img.get(img_info['id'], []):
            cid = ann['category_id']
            color = COLORS[(cid - 1) % len(COLORS)]
            mask = decode_rle(ann['segmentation'], h, w)
            if mask is not None:
                vis[mask > 0] = (vis[mask > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            x, y, bw, bh = [int(v) for v in ann['bbox']]
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 1)
            cv2.putText(vis, cat_names.get(cid, str(cid)), (x, max(y - 2, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        out_path = out_dir / f"{Path(img_info['file_name']).stem}_inst.jpg"
        Image.fromarray(vis).save(out_path)
        print(f'Wrote {out_path} ({len(anns_by_img.get(img_info["id"], []))} instances)')

    print(f'Done. {len(sample)} images -> {out_dir}')


if __name__ == '__main__':
    main()
