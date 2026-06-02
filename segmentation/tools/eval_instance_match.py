#!/usr/bin/env python3
"""Segmentation-level instance match between pred/GT COCO JSON (F1-4)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transgrasp.pipelines.roi_postprocess import CLASSES, mask_iou


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred-coco', type=str, required=True)
    p.add_argument('--gt-coco', type=str, required=True)
    p.add_argument('--iou-match', type=float, default=0.25)
    p.add_argument('--out', type=str, required=True)
    return p.parse_args()


def resolve(p: str) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def decode_masks(coco: dict) -> dict[int, list[dict]]:
    from pycocotools import mask as mask_utils

    anns_by_img: dict[int, list] = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_img[ann['image_id']].append(ann)

    out: dict[int, list[dict]] = {}
    for img_id, anns in anns_by_img.items():
        rows = []
        for ann in anns:
            seg = ann['segmentation']
            if isinstance(seg, dict):
                mask = mask_utils.decode(seg)
            else:
                h = next(
                    im['height'] for im in coco['images'] if im['id'] == img_id)
                w = next(
                    im['width'] for im in coco['images'] if im['id'] == img_id)
                rle = mask_utils.frPyObjects(seg, h, w)
                mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                mask = mask[..., 0]
            rows.append({
                'category_id': ann['category_id'],
                'class_name': CLASSES[ann['category_id']],
                'mask': (mask > 0).astype(np.uint8),
                'score': ann.get('score', 1.0),
            })
        out[img_id] = rows
    return out


def greedy_match(
    preds: list[dict], gts: list[dict], iou_thresh: float,
) -> tuple[int, int, int]:
    used = set()
    matched = 0
    for gt in gts:
        best_pi, best_iou = -1, 0.0
        for pi, pred in enumerate(preds):
            if pi in used:
                continue
            iou = mask_iou(gt['mask'], pred['mask'])
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        if best_pi >= 0 and best_iou >= iou_thresh:
            used.add(best_pi)
            matched += 1
    return matched, len(gts), len(preds)


def main():
    args = parse_args()
    pred_coco = json.loads(resolve(args.pred_coco).read_text(encoding='utf-8'))
    gt_coco = json.loads(resolve(args.gt_coco).read_text(encoding='utf-8'))

    pred_by_img = decode_masks(pred_coco)
    gt_by_img = decode_masks(gt_coco)

    total_gt = 0
    total_pred = 0
    total_matched = 0
    per_class_gt = defaultdict(int)
    per_class_matched = defaultdict(int)

    for img_id, gts in gt_by_img.items():
        preds = pred_by_img.get(img_id, [])
        matched, n_gt, n_pred = greedy_match(preds, gts, args.iou_match)
        total_gt += n_gt
        total_pred += n_pred
        total_matched += matched
        for gt in gts:
            per_class_gt[gt['class_name']] += 1
        # attribute matches to GT class
        used = set()
        for gt in gts:
            best_pi, best_iou = -1, 0.0
            for pi, pred in enumerate(preds):
                if pi in used:
                    continue
                iou = mask_iou(gt['mask'], pred['mask'])
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi
            if best_pi >= 0 and best_iou >= args.iou_match:
                used.add(best_pi)
                per_class_matched[gt['class_name']] += 1

    per_class = {}
    for name in sorted(per_class_gt.keys()):
        gt_n = per_class_gt[name]
        m_n = per_class_matched.get(name, 0)
        per_class[name] = {
            'gt_instances': gt_n,
            'matched': m_n,
            'match_rate': round(m_n / max(gt_n, 1), 4),
            'miss': gt_n - m_n,
        }

    result = {
        'match_rate': round(total_matched / max(total_gt, 1), 4),
        'pred_gt_ratio': round(total_pred / max(total_gt, 1), 4),
        'num_gt': total_gt,
        'num_pred': total_pred,
        'num_matched': total_matched,
        'iou_match': args.iou_match,
        'per_class': per_class,
    }

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'-> {out_path}')


if __name__ == '__main__':
    main()
