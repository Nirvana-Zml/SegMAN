#!/usr/bin/env python3
"""Export Mask2Former val predictions to COCO instance JSON (F1-4)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transgrasp.pipelines.roi_postprocess import CLASSES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--ann', type=str,
                   default='segmentation/data/trans10k/coco_instances/val.json')
    p.add_argument('--img-root', type=str, default='segmentation/data/trans10k')
    p.add_argument('--out-dir', type=str, required=True)
    p.add_argument('--score-thresh', type=float, default=0.3)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--max-images', type=int, default=0)
    return p.parse_args()


def resolve(p: str) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def encode_rle(binary: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils
    binary = np.asfortranarray(binary.astype(np.uint8))
    rle = mask_utils.encode(binary)
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def mask_to_bbox(binary: np.ndarray) -> list[float]:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    return [x0, y0, x1 - x0, y1 - y0]


def extract_instances(result, score_thresh: float) -> list[dict]:
    """Parse mmdet 3.x InstanceData / tuple output."""
    from mmdet.structures import DetDataSample

    if isinstance(result, DetDataSample):
        pred = result.pred_instances
        scores = pred.scores.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        masks = pred.masks.cpu().numpy()
    elif isinstance(result, (list, tuple)) and len(result) >= 2:
        # legacy (bbox, segm) tuple
        _, segm = result[0], result[1]
        out = []
        for cat_id, segs in segm.items():
            for seg in segs:
                score = float(seg.get('score', 1.0))
                if score < score_thresh:
                    continue
                mask = seg['mask']
                out.append({'category_id': int(cat_id), 'score': score, 'mask': mask})
        return out
    else:
        pred = result.pred_instances
        scores = pred.scores.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        masks = pred.masks.cpu().numpy()

    instances = []
    for score, label, mask in zip(scores, labels, masks):
        if float(score) < score_thresh:
            continue
        if mask.ndim == 3:
            mask = mask[0]
        binary = (mask > 0.5).astype(np.uint8)
        if binary.sum() == 0:
            continue
        cat_id = int(label) + 1  # mmdet 0-indexed label -> COCO category_id
        if cat_id <= 0 or cat_id >= len(CLASSES):
            continue
        instances.append({
            'category_id': cat_id,
            'score': float(score),
            'mask': binary,
        })
    return instances


def main():
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = resolve(args.img_root)

    try:
        from mmdet.apis import init_detector, inference_detector
    except ImportError as exc:
        raise SystemExit(
            'mmdet required. Run: bash scripts/setup_f1_mmdet_env.sh'
        ) from exc

    gt = json.loads(resolve(args.ann).read_text(encoding='utf-8'))
    images = gt['images']
    if args.max_images > 0:
        images = images[: args.max_images]

    model = init_detector(
        str(resolve(args.config)),
        str(resolve(args.checkpoint)),
        device=args.device,
    )

    pred_images = []
    pred_anns = []
    ann_id = 1

    for img_info in tqdm(images, desc='m2f infer'):
        img_id = img_info['id']
        img_dir = img_root / 'img_dir' / 'val'
        img_path = img_dir / img_info['file_name']
        if not img_path.is_file():
            stem = Path(img_info['file_name']).stem
            for ext in ('.jpg', '.png'):
                alt = img_dir / f'{stem}{ext}'
                if alt.is_file():
                    img_path = alt
                    break

        pred_images.append({
            'id': img_id,
            'file_name': img_info['file_name'],
            'width': img_info['width'],
            'height': img_info['height'],
        })

        result = inference_detector(model, str(img_path))
        for inst in extract_instances(result, args.score_thresh):
            binary = inst['mask']
            pred_anns.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': inst['category_id'],
                'bbox': mask_to_bbox(binary),
                'area': float(binary.sum()),
                'segmentation': encode_rle(binary),
                'iscrowd': 0,
                'score': inst['score'],
            })
            ann_id += 1

    out_json = {
        'images': pred_images,
        'annotations': pred_anns,
        'categories': gt.get('categories', []),
    }
    out_path = out_dir / 'pred_instances.json'
    out_path.write_text(json.dumps(out_json, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(pred_anns)} instances -> {out_path}')


if __name__ == '__main__':
    main()
