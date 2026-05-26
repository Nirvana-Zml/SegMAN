#!/usr/bin/env python3
"""E2E: full image -> SegMAN semantic seg -> ROI crops -> OpenCLIP classify (+ reject).

Examples:
  # Single image
  python transgrasp/pipelines/segment_and_classify.py \\
    --image segmentation/data/trans10k/img_dir/val/val_000000.jpg

  # Val set benchmark (match pred instances to GT by mask IoU)
  python transgrasp/pipelines/segment_and_classify.py \\
    --eval-split val --max-images 50 --out-dir outputs/e2e_test/val50
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from transgrasp.classification.eval_openclip_classifier import build_from_checkpoint
from transgrasp.classification.eval_reject_policy import load_class_thresholds
from transgrasp.pipelines.classify_instances import classify_instances
from transgrasp.pipelines.roi_extract import extract_instance_rois, mask_iou
from transgrasp.pipelines.seg_model import (
    DEFAULT_SEG_CHECKPOINT,
    DEFAULT_SEG_CONFIG,
    SegMANSegmentor,
)


def parse_args():
    p = argparse.ArgumentParser(description='E2E segment + classify on Trans10K images')
    p.add_argument('--image', type=str, default=None, help='Single image path')
    p.add_argument('--image-dir', type=str, default=None, help='Directory of images (*.jpg/png)')
    p.add_argument('--eval-split', choices=['train', 'val'], default=None,
                   help='Run on MMSeg Trans10K split with GT matching metrics')
    p.add_argument('--data-root', type=str, default=None,
                   help='MMSeg root (default: segmentation/data/trans10k)')
    p.add_argument('--seg-config', type=str, default=DEFAULT_SEG_CONFIG)
    p.add_argument('--seg-checkpoint', type=str, default=DEFAULT_SEG_CHECKPOINT)
    p.add_argument('--cls-checkpoint', type=str,
                   default='outputs/openclip_classifier/deliver_classifier_best.pth')
    p.add_argument('--class-thresholds', type=str,
                   default='transgrasp/classification/configs/reject_thresholds_p3.json')
    p.add_argument('--out-dir', type=str, default='outputs/e2e_segment_classify')
    p.add_argument('--bbox-pad', type=float, default=0.15)
    p.add_argument('--min-area', type=int, default=64)
    p.add_argument('--iou-match', type=float, default=0.3,
                   help='Min mask IoU to match pred instance to GT (eval mode)')
    p.add_argument('--max-images', type=int, default=-1)
    p.add_argument('--save-rois', action='store_true', help='Save ROI crop JPGs per image')
    p.add_argument('--save-sem-seg', action='store_true', help='Save predicted class-id PNG')
    p.add_argument('--device', type=str, default='cuda:0')
    return p.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def resolve_mmseg_root(data_root: str | None) -> Path:
    candidates = []
    if data_root:
        candidates.append(resolve_path(data_root))
    candidates.extend([
        PROJECT_ROOT / 'segmentation' / 'data' / 'trans10k',
        PROJECT_ROOT / 'data' / 'trans10k',
    ])
    for c in candidates:
        if (c / 'img_dir').is_dir() and (c / 'ann_dir').is_dir():
            return c
    raise FileNotFoundError(
        'Cannot find MMSeg trans10k (img_dir/ + ann_dir/). Use --data-root.')


def iter_split_images(mmseg_root: Path, split: str):
    img_dir = mmseg_root / 'img_dir' / split
    ann_dir = mmseg_root / 'ann_dir' / split
    stems = sorted({p.stem for p in img_dir.glob('*.jpg')} | {p.stem for p in img_dir.glob('*.png')})
    for stem in stems:
        img_path = img_dir / f'{stem}.jpg'
        if not img_path.is_file():
            img_path = img_dir / f'{stem}.png'
        ann_path = ann_dir / f'{stem}.png'
        if img_path.is_file() and ann_path.is_file():
            yield stem, img_path, ann_path


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'))


def load_gt_label(ann_path: Path) -> np.ndarray:
    label = np.array(Image.open(ann_path))
    if label.ndim == 3:
        label = label[..., 0]
    return label.astype(np.uint8)


def match_instances_to_gt(
    pred_rows: list[dict],
    pred_instances,
    gt_instances,
    iou_thresh: float,
) -> list[dict]:
    """Greedy IoU matching: each GT matched to at most one pred."""
    used_pred = set()
    matches = []
    for gi, gt in enumerate(gt_instances):
        best_pi, best_iou = -1, 0.0
        for pi, pred in enumerate(pred_instances):
            if pi in used_pred:
                continue
            iou = mask_iou(gt.mask, pred.mask)
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        m = {
            'gt_class': gt.class_name,
            'gt_class_id': gt.class_id,
            'match_iou': round(best_iou, 4),
            'matched': best_iou >= iou_thresh and best_pi >= 0,
        }
        if m['matched']:
            used_pred.add(best_pi)
            pr = pred_rows[best_pi]
            m.update({
                'pred_class': pr['pred_class'],
                'confidence': pr['confidence'],
                'action': pr['action'],
                'correct': pr['pred_class'] == gt.class_name,
                'seg_class': pred_instances[best_pi].class_name,
            })
        matches.append(m)
    return matches


def process_one_image(
    stem: str,
    rgb: np.ndarray,
    segmentor: SegMANSegmentor,
    cls_model,
    preprocess,
    class_names: list[str],
    class_thresholds: dict[str, float],
    device: torch.device,
    args,
    gt_label: np.ndarray | None = None,
) -> dict:
    pred_label = segmentor.predict_label_map(rgb)
    pred_instances = extract_instance_rois(
        rgb, pred_label, bbox_pad=args.bbox_pad, min_area=args.min_area)
    pred_rows = classify_instances(
        cls_model, preprocess, class_names, class_thresholds,
        pred_instances, device)

    out = {
        'image_stem': stem,
        'num_pred_instances': len(pred_rows),
        'instances': pred_rows,
    }

    if gt_label is not None:
        gt_instances = extract_instance_rois(
            rgb, gt_label, bbox_pad=args.bbox_pad, min_area=args.min_area)
        matches = match_instances_to_gt(
            pred_rows, pred_instances, gt_instances, args.iou_match)
        n_gt = len(gt_instances)
        n_matched = sum(1 for m in matches if m['matched'])
        n_correct = sum(1 for m in matches if m.get('correct'))
        n_grasp = sum(
            1 for m in matches
            if m['matched'] and m.get('action') == 'grasp')
        n_correct_grasp = sum(
            1 for m in matches
            if m['matched'] and m.get('action') == 'grasp' and m.get('correct'))
        out['eval'] = {
            'num_gt_instances': n_gt,
            'num_pred_instances': len(pred_instances),
            'num_matched': n_matched,
            'match_rate': round(n_matched / max(n_gt, 1), 4),
            'top1_on_matched': round(n_correct / max(n_matched, 1), 4),
            'top1_on_matched_grasp_only': round(
                n_correct_grasp / max(n_grasp, 1), 4),
            'accept_rate_on_matched': round(n_grasp / max(n_matched, 1), 4),
            'matches': matches,
        }

    return out, pred_label, pred_instances


def save_image_artifacts(
    out_dir: Path,
    stem: str,
    pred_label: np.ndarray,
    pred_instances,
    pred_rows: list[dict],
    args,
):
    if args.save_sem_seg:
        sem_dir = out_dir / 'sem_seg_pred'
        sem_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred_label).save(sem_dir / f'{stem}.png')

    if args.save_rois:
        roi_dir = out_dir / 'roi_crops' / stem
        roi_dir.mkdir(parents=True, exist_ok=True)
        for i, (inst, row) in enumerate(zip(pred_instances, pred_rows)):
            fname = (
                f"{row['pred_class']}_{row['action']}_"
                f"{row['confidence']:.2f}_{i:03d}.jpg"
            )
            Image.fromarray(inst.crop_rgb).save(roi_dir / fname)


def aggregate_eval(results: list[dict]) -> dict:
    total_gt = total_matched = total_correct = 0
    total_grasp = total_correct_grasp = 0
    for r in results:
        ev = r.get('eval')
        if not ev:
            continue
        total_gt += ev['num_gt_instances']
        total_matched += ev['num_matched']
        for m in ev['matches']:
            if m['matched']:
                if m.get('correct'):
                    total_correct += 1
                if m.get('action') == 'grasp':
                    total_grasp += 1
                    if m.get('correct'):
                        total_correct_grasp += 1
    return {
        'num_images': len(results),
        'num_gt_instances': total_gt,
        'num_matched': total_matched,
        'match_rate': round(total_matched / max(total_gt, 1), 4),
        'e2e_top1_on_matched': round(total_correct / max(total_matched, 1), 4),
        'e2e_top1_grasp_only': round(total_correct_grasp / max(total_grasp, 1), 4),
        'grasp_rate_on_matched': round(total_grasp / max(total_matched, 1), 4),
    }


def main():
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        device_str = 'cpu'
    device = torch.device(
        'cuda' if device_str.startswith('cuda') and torch.cuda.is_available() else 'cpu')

    print('Loading SegMAN...')
    segmentor = SegMANSegmentor(
        args.seg_config, args.seg_checkpoint, device=device_str)

    print('Loading OpenCLIP classifier...')
    ckpt = resolve_path(args.cls_checkpoint)
    thresh_path = resolve_path(args.class_thresholds)
    cls_model, preprocess, class_names, cls_meta = build_from_checkpoint(ckpt, device)
    class_thresholds = load_class_thresholds(thresh_path, class_names)

    meta = {
        'seg_config': args.seg_config,
        'seg_checkpoint': args.seg_checkpoint,
        'cls_checkpoint': str(ckpt),
        'class_thresholds': str(thresh_path),
        'bbox_pad': args.bbox_pad,
        'min_area': args.min_area,
        'iou_match': args.iou_match,
    }

    results = []

    if args.image:
        img_path = resolve_path(args.image)
        stem = img_path.stem
        rgb = load_rgb(img_path)
        one, pred_label, pred_instances = process_one_image(
            stem, rgb, segmentor, cls_model, preprocess, class_names,
            class_thresholds, device, args, gt_label=None)
        save_image_artifacts(out_dir, stem, pred_label, pred_instances, one['instances'], args)
        out_path = out_dir / f'{stem}.json'
        out_path.write_text(json.dumps(one, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps(one, indent=2, ensure_ascii=False))
        print(f'Wrote {out_path}')
        return

    pairs = []
    if args.eval_split:
        mmseg_root = resolve_mmseg_root(args.data_root)
        pairs = list(iter_split_images(mmseg_root, args.eval_split))
        meta['data_root'] = str(mmseg_root)
        meta['split'] = args.eval_split
    elif args.image_dir:
        img_dir = resolve_path(args.image_dir)
        for p in sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png')):
            pairs.append((p.stem, p, None))
    else:
        print('Error: specify --image, --image-dir, or --eval-split', file=sys.stderr)
        sys.exit(1)

    if args.max_images >= 0:
        pairs = pairs[: args.max_images]

    print(f'Processing {len(pairs)} image(s)...')
    for stem, img_path, ann_path in pairs:
        rgb = load_rgb(img_path)
        gt_label = load_gt_label(ann_path) if ann_path is not None else None
        one, pred_label, pred_instances = process_one_image(
            stem, rgb, segmentor, cls_model, preprocess, class_names,
            class_thresholds, device, args, gt_label=gt_label)
        save_image_artifacts(out_dir, stem, pred_label, pred_instances, one['instances'], args)
        per_path = out_dir / 'per_image' / f'{stem}.json'
        per_path.parent.mkdir(parents=True, exist_ok=True)
        per_path.write_text(json.dumps(one, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        results.append(one)
        if 'eval' in one:
            ev = one['eval']
            print(
                f"  {stem}: pred={ev['num_pred_instances']} gt={ev['num_gt_instances']} "
                f"matched={ev['num_matched']} acc={ev['top1_on_matched']:.4f}")

    summary = {'meta': meta, 'per_image': [r['image_stem'] for r in results]}
    if args.eval_split:
        summary['aggregate'] = aggregate_eval(results)
        print('\n=== E2E aggregate (GT instance matching) ===')
        print(json.dumps(summary['aggregate'], indent=2))

    summary_path = out_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    if args.eval_split:
        csv_path = out_dir / 'per_image_metrics.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                'stem', 'num_gt', 'num_pred', 'num_matched', 'match_rate',
                'top1_matched', 'top1_grasp', 'accept_rate',
            ])
            for r in results:
                ev = r['eval']
                w.writerow([
                    r['image_stem'],
                    ev['num_gt_instances'],
                    ev['num_pred_instances'],
                    ev['num_matched'],
                    ev['match_rate'],
                    ev['top1_on_matched'],
                    ev['top1_on_matched_grasp_only'],
                    ev['accept_rate_on_matched'],
                ])

    print(f'\nSummary -> {summary_path}')


if __name__ == '__main__':
    main()
