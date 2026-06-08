#!/usr/bin/env python3
"""Build OpenCLIP ROI dataset from Mask2Former instances matched to GT (F1 P1).

For each image: M2F predict -> greedy match to GT instances (mask IoU).
Matched pred crops are saved with **GT class** labels (same distribution as E2E cls_on_matched).

Output layout (aligned with ROIDataset):
  {out_root}/meta/classes.txt
  {out_root}/{split}/images/*.jpg
  {out_root}/{split}/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transgrasp.pipelines.instance_predictor import build_instance_predictor
from transgrasp.pipelines.roi_postprocess import CLASSES, FOREGROUND_IDS
from transgrasp.pipelines.segment_and_classify import (
    build_extract_config,
    build_gt_extract_config,
    iter_split_images,
    load_gt_label,
    load_rgb,
    match_instances_to_gt,
    parse_iou_match_per_class,
    resolve_mmseg_root,
    resolve_path,
)
from transgrasp.pipelines.roi_extract import extract_instance_rois, mask_iou


def parse_args():
    p = argparse.ArgumentParser(description='Export M2F-matched GT-labeled ROI dataset')
    p.add_argument('--split', choices=['train', 'val'], required=True)
    p.add_argument('--out-root', type=str, default='data/trans10k_roi_m2f')
    p.add_argument('--data-root', type=str, default=None)
    p.add_argument('--m2f-config', type=str,
                   default='segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py')
    p.add_argument('--m2f-checkpoint', type=str,
                   default='segmentation/outputs/m2f_trans10k_pseudo/iter_40000.pth')
    p.add_argument('--m2f-score-thresh', type=float, default=0.30)
    p.add_argument('--bbox-pad', type=float, default=0.15)
    p.add_argument('--min-area', type=int, default=128)
    p.add_argument('--nms-iou', type=float, default=0.5)
    p.add_argument('--iou-match', type=float, default=0.25)
    p.add_argument('--iou-match-per-class', type=str, default='')
    p.add_argument('--match-algorithm', choices=['greedy', 'hungarian'], default='greedy')
    p.add_argument('--include-unmatched', action='store_true',
                   help='Also export unmatched preds with M2F seg class label (default: matched only)')
    p.add_argument('--max-images', type=int, default=-1)
    p.add_argument('--device', type=str, default='cuda:0')
    return p.parse_args()


def write_meta(out_root: Path):
    meta_dir = out_root / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)
    classes_path = meta_dir / 'classes.txt'
    names = [CLASSES[i] for i in FOREGROUND_IDS]
    classes_path.write_text('\n'.join(names) + '\n', encoding='utf-8')


def export_split(args, mmseg_root: Path, split: str, out_root: Path, predictor) -> dict:
    split_dir = out_root / split
    images_dir = split_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    e2e_args = Namespace(
        bbox_pad=args.bbox_pad,
        min_area=args.min_area,
        min_area_shelf=0,
        max_aspect_ratio=10.0,
        nms_iou=args.nms_iou,
        merge_cc_iou=0.0,
        merge_cc_dist=8,
        merge_cc_classes='',
        iou_match=args.iou_match,
        iou_match_per_class=args.iou_match_per_class,
        match_algorithm=args.match_algorithm,
    )
    ext_cfg = build_extract_config(e2e_args)
    gt_cfg = build_gt_extract_config(e2e_args)
    per_class_iou = parse_iou_match_per_class(args.iou_match_per_class)

    rows = []
    counters = {cid: 0 for cid in FOREGROUND_IDS}
    stats = {
        'split': split,
        'images': 0,
        'gt_instances': 0,
        'pred_instances': 0,
        'matched_exports': 0,
        'unmatched_exports': 0,
    }

    pairs = list(iter_split_images(mmseg_root, split))
    if args.max_images >= 0:
        pairs = pairs[: args.max_images]

    for stem, img_path, ann_path in pairs:
        rgb = load_rgb(img_path)
        gt_label = load_gt_label(ann_path)
        h, w = rgb.shape[:2]
        if gt_label.shape[:2] != (h, w):
            gt_label = np.array(
                Image.fromarray(gt_label).resize((w, h), Image.NEAREST))

        pred_instances = predictor.predict_instances(rgb, ext_cfg)
        gt_instances = extract_instance_rois(rgb, gt_label, extract_cfg=gt_cfg)
        pred_rows = [
            {
                'pred_class': p.class_name,
                'confidence': 1.0,
                'action': 'grasp',
            }
            for p in pred_instances
        ]
        matches = match_instances_to_gt(
            pred_rows, pred_instances, gt_instances,
            args.iou_match, algorithm=args.match_algorithm,
            per_class_iou=per_class_iou)

        stats['images'] += 1
        stats['gt_instances'] += len(gt_instances)
        stats['pred_instances'] += len(pred_instances)

        used_pred: set[int] = set()
        for gt, m in zip(gt_instances, matches):
            if not m.get('matched'):
                continue
            thresh = args.iou_match
            if args.iou_match_per_class:
                per = parse_iou_match_per_class(args.iou_match_per_class)
                thresh = per.get(gt.class_name, args.iou_match)
            best_pi, best_iou = -1, 0.0
            for pi, pred in enumerate(pred_instances):
                if pi in used_pred:
                    continue
                iou = mask_iou(gt.mask, pred.mask)
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi
            if best_pi < 0 or best_iou < thresh:
                continue
            used_pred.add(best_pi)
            pred = pred_instances[best_pi]
            class_name = gt.class_name
            class_id = gt.class_id
            idx = counters[class_id]
            counters[class_id] += 1
            fname = f'{split}_{stem}_{class_name}_{idx:05d}.jpg'
            out_path = images_dir / fname
            Image.fromarray(pred.crop_rgb).save(out_path, quality=95)
            rows.append({
                'path': f'images/{fname}',
                'class_id': class_id,
                'class_name': class_name,
                'src_image': img_path.name,
                'instance_id': pred.instance_id,
                'mask_source': 'm2f_matched_gt',
            })
            stats['matched_exports'] += 1

        if args.include_unmatched:
            for pi, pred in enumerate(pred_instances):
                if pi in used_pred:
                    continue
                class_id = pred.class_id
                class_name = pred.class_name
                idx = counters[class_id]
                counters[class_id] += 1
                fname = f'{split}_{stem}_unmatched_{class_name}_{idx:05d}.jpg'
                out_path = images_dir / fname
                Image.fromarray(pred.crop_rgb).save(out_path, quality=95)
                rows.append({
                    'path': f'images/{fname}',
                    'class_id': class_id,
                    'class_name': class_name,
                    'src_image': img_path.name,
                    'instance_id': pred.instance_id,
                    'mask_source': 'm2f_unmatched',
                })
                stats['unmatched_exports'] += 1

    labels_path = split_dir / 'labels.csv'
    with labels_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['path', 'class_id', 'class_name', 'src_image', 'instance_id', 'mask_source'],
        )
        writer.writeheader()
        writer.writerows(rows)

    stats['total_rois'] = len(rows)
    stats['labels_csv'] = str(labels_path)
    print(f'[{split}] {stats["images"]} images -> {len(rows)} ROIs ({stats["matched_exports"]} matched)')
    return stats


def main():
    args = parse_args()
    out_root = resolve_path(args.out_root)
    mmseg_root = resolve_mmseg_root(args.data_root)
    write_meta(out_root)

    pred_args = Namespace(
        instance_source='m2f',
        m2f_config=str(resolve_path(args.m2f_config)),
        m2f_checkpoint=str(resolve_path(args.m2f_checkpoint)),
        m2f_score_thresh=args.m2f_score_thresh,
        device=args.device,
        maskrcnn_checkpoint='',
        refine_morph_close=0,
        refine_morph_classes='',
        refine_dilate='',
        refine_erode='',
        refine_crf=False,
        seg_tta=False,
    )
    predictor = build_instance_predictor(pred_args)

    stats = export_split(args, mmseg_root, args.split, out_root, predictor)
    manifest_path = out_root / 'meta' / f'manifest_{args.split}.json'
    manifest_path.write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
