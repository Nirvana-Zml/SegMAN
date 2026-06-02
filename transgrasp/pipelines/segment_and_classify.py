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
from transgrasp.pipelines.roi_postprocess import ExtractConfig
from transgrasp.pipelines.instance_predictor import build_instance_predictor
from transgrasp.pipelines.seg_refine import apply_seg_refine, build_refine_config
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
    p.add_argument('--min-area-shelf', type=int, default=0,
                   help='Per-class min_area for shelf (0=use --min-area)')
    p.add_argument('--max-aspect-ratio', type=float, default=10.0,
                   help='Drop CC with aspect ratio above this (0=disable)')
    p.add_argument('--nms-iou', type=float, default=0.0,
                   help='Same-class bbox NMS IoU threshold (0=disable)')
    p.add_argument('--merge-cc-iou', type=float, default=0.0,
                   help='Merge same-class CC if bbox IoU >= this (0=disable, B5)')
    p.add_argument('--merge-cc-dist', type=int, default=8,
                   help='Merge same-class CC if bbox edge gap <= px (B5)')
    p.add_argument('--merge-cc-classes', type=str, default='',
                   help='Comma-separated classes to merge (empty=all foreground)')
    p.add_argument('--iou-match', type=float, default=0.3,
                   help='Min mask IoU to match pred instance to GT (eval mode)')
    p.add_argument('--iou-match-per-class', type=str, default='',
                   help='Per-class IoU thresholds, e.g. door:0.25,wall:0.25,cup:0.35 (B3)')
    p.add_argument('--match-algorithm', choices=['greedy', 'hungarian'], default='greedy',
                   help='GT-pred instance matching algorithm (B2)')
    p.add_argument('--max-images', type=int, default=-1)
    p.add_argument('--save-rois', action='store_true', help='Save ROI crop JPGs per image')
    p.add_argument('--save-sem-seg', action='store_true', help='Save predicted class-id PNG')
    p.add_argument('--device', type=str, default='cuda:0')
    # Scheme D: inference-side mask refine
    p.add_argument('--refine-morph-close', type=int, default=0,
                   help='Morph close kernel size (0=off, D1)')
    p.add_argument('--refine-morph-classes', type=str, default='',
                   help='Classes for morph close, e.g. wall,door,window')
    p.add_argument('--refine-dilate', type=str, default='',
                   help='Per-class dilate px, e.g. wall:2,door:2,window:1 (D2)')
    p.add_argument('--refine-erode', type=str, default='',
                   help='Per-class erode px, e.g. shelf:1 (D2)')
    p.add_argument('--refine-crf', action='store_true', help='Dense CRF refine (D3)')
    p.add_argument('--refine-crf-iters', type=int, default=5)
    p.add_argument('--refine-crf-classes', type=str, default='',
                   help='Only apply CRF output on these classes')
    p.add_argument('--refine-split-door-wall', action='store_true',
                   help='Peel door pixels adjacent to wall (D5)')
    p.add_argument('--seg-tta', action='store_true',
                   help='Multi-scale + flip TTA fusion (D4)')
    p.add_argument('--seg-tta-scales', type=str, default='0.75,1.0,1.25')
    # Scheme E: instance source
    p.add_argument('--instance-source', choices=['semantic', 'gt_oracle', 'maskrcnn', 'm2f'],
                   default='semantic',
                   help='Instance mask source (E4/F1): semantic / GT oracle / Mask R-CNN / Mask2Former')
    p.add_argument('--maskrcnn-checkpoint', type=str,
                   default='segmentation/outputs/maskrcnn_trans10k_pseudo/best.pth')
    p.add_argument('--m2f-config', type=str,
                   default='segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py')
    p.add_argument('--m2f-checkpoint', type=str,
                   default='segmentation/outputs/m2f_trans10k_pseudo/best_bbox_mAP.pth')
    p.add_argument('--m2f-score-thresh', type=float, default=0.3)
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


def parse_iou_match_per_class(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not spec.strip():
        return out
    for part in spec.split(','):
        part = part.strip()
        if not part or ':' not in part:
            continue
        name, val = part.split(':', 1)
        out[name.strip()] = float(val.strip())
    return out


def build_extract_config(args) -> ExtractConfig:
    per_class = {}
    if args.min_area_shelf > 0:
        per_class['shelf'] = args.min_area_shelf
    merge_classes = tuple(
        c.strip() for c in args.merge_cc_classes.split(',') if c.strip())
    return ExtractConfig(
        bbox_pad=args.bbox_pad,
        min_area=args.min_area,
        min_area_per_class=per_class,
        max_aspect_ratio=args.max_aspect_ratio,
        nms_iou=args.nms_iou,
        enable_nms=args.nms_iou > 0,
        merge_cc_iou=args.merge_cc_iou,
        merge_cc_dist=args.merge_cc_dist,
        merge_cc_classes=merge_classes,
    )


def build_gt_extract_config(args) -> ExtractConfig:
    """GT instances use fixed baseline rules (min_area=64, no NMS) for fair match_rate."""
    return ExtractConfig(
        bbox_pad=args.bbox_pad,
        min_area=64,
        max_aspect_ratio=0.0,
        enable_nms=False,
    )


def _iou_thresh_for_gt(gt_class: str, default: float, per_class: dict[str, float]) -> float:
    return per_class.get(gt_class, default)


def _build_match_record(
    gt,
    best_pi: int,
    best_iou: float,
    overlap_count: int,
    matched: bool,
    pred_rows,
    pred_instances,
) -> dict:
    m = {
        'gt_class': gt.class_name,
        'gt_class_id': gt.class_id,
        'match_iou': round(best_iou, 4),
        'pred_overlap_count': overlap_count,
        'matched': matched,
    }
    if matched and best_pi >= 0:
        pr = pred_rows[best_pi]
        m.update({
            'pred_class': pr['pred_class'],
            'confidence': pr['confidence'],
            'action': pr['action'],
            'correct': pr['pred_class'] == gt.class_name,
            'seg_class': pred_instances[best_pi].class_name,
        })
    elif best_pi >= 0:
        pr = pred_rows[best_pi]
        m['best_seg_class'] = pred_instances[best_pi].class_name
        m['best_pred_class'] = pr['pred_class']
    return m


def match_instances_greedy(
    pred_rows, pred_instances, gt_instances,
    iou_thresh: float, per_class_iou: dict[str, float],
) -> list[dict]:
    used_pred = set()
    matches = []
    for gt in gt_instances:
        thresh = _iou_thresh_for_gt(gt.class_name, iou_thresh, per_class_iou)
        best_pi, best_iou = -1, 0.0
        overlap_count = 0
        for pi, pred in enumerate(pred_instances):
            iou = mask_iou(gt.mask, pred.mask)
            if iou > 0.05:
                overlap_count += 1
            if pi in used_pred:
                continue
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        matched = best_iou >= thresh and best_pi >= 0
        if matched:
            used_pred.add(best_pi)
        matches.append(_build_match_record(
            gt, best_pi, best_iou, overlap_count, matched,
            pred_rows, pred_instances))
    return matches


def match_instances_hungarian(
    pred_rows, pred_instances, gt_instances,
    iou_thresh: float, per_class_iou: dict[str, float],
) -> list[dict]:
    from scipy.optimize import linear_sum_assignment

    n_gt = len(gt_instances)
    n_pred = len(pred_instances)
    if n_gt == 0:
        return []
    if n_pred == 0:
        return [_build_match_record(gt, -1, 0.0, 0, False, pred_rows, pred_instances)
                for gt in gt_instances]

    big = 1e6
    cost = np.full((n_gt, n_pred), big, dtype=np.float64)
    iou_mat = np.zeros((n_gt, n_pred), dtype=np.float64)
    for gi, gt in enumerate(gt_instances):
        for pi, pred in enumerate(pred_instances):
            iou = mask_iou(gt.mask, pred.mask)
            iou_mat[gi, pi] = iou
            cost[gi, pi] = 1.0 - iou

    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    matches = []
    for gi, gt in enumerate(gt_instances):
        thresh = _iou_thresh_for_gt(gt.class_name, iou_thresh, per_class_iou)
        overlap_count = int((iou_mat[gi] > 0.05).sum())
        pi = assignment.get(gi, -1)
        best_iou = float(iou_mat[gi, pi]) if pi >= 0 else 0.0
        matched = pi >= 0 and best_iou >= thresh
        matches.append(_build_match_record(
            gt, pi if pi >= 0 else -1, best_iou, overlap_count, matched,
            pred_rows, pred_instances))
    return matches


def match_instances_to_gt(
    pred_rows: list[dict],
    pred_instances,
    gt_instances,
    iou_thresh: float,
    algorithm: str = 'greedy',
    per_class_iou: dict[str, float] | None = None,
) -> list[dict]:
    per_class_iou = per_class_iou or {}
    if algorithm == 'hungarian':
        return match_instances_hungarian(
            pred_rows, pred_instances, gt_instances, iou_thresh, per_class_iou)
    return match_instances_greedy(
        pred_rows, pred_instances, gt_instances, iou_thresh, per_class_iou)


def parse_tta_scales(spec: str) -> tuple[float, ...]:
    if not spec.strip():
        return (0.75, 1.0, 1.25)
    return tuple(float(x.strip()) for x in spec.split(',') if x.strip())


def process_one_image(
    stem: str,
    rgb: np.ndarray,
    segmentor: SegMANSegmentor | None,
    cls_model,
    preprocess,
    class_names: list[str],
    class_thresholds: dict[str, float],
    device: torch.device,
    args,
    gt_label: np.ndarray | None = None,
    refine_cfg=None,
    instance_predictor=None,
) -> dict:
    ext_cfg = build_extract_config(args)
    pred_label = None

    if args.instance_source == 'semantic':
        if segmentor is None:
            raise ValueError('segmentor required for semantic instance source')
        if args.seg_tta:
            pred_label = segmentor.predict_label_map_tta(
                rgb, scales=parse_tta_scales(args.seg_tta_scales))
        else:
            pred_label = segmentor.predict_label_map(rgb)
        if refine_cfg is not None:
            pred_label = apply_seg_refine(rgb, pred_label, refine_cfg)
        pred_instances = extract_instance_rois(rgb, pred_label, extract_cfg=ext_cfg)
    else:
        if instance_predictor is None:
            instance_predictor = build_instance_predictor(
                args, segmentor=segmentor, gt_label=gt_label)
        pred_instances = instance_predictor.predict_instances(rgb, ext_cfg)
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
            rgb, gt_label, extract_cfg=build_gt_extract_config(args))
        matches = match_instances_to_gt(
            pred_rows, pred_instances, gt_instances, args.iou_match,
            algorithm=args.match_algorithm,
            per_class_iou=parse_iou_match_per_class(args.iou_match_per_class))
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
        if pred_label is None:
            return
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
    total_gt = total_pred = total_matched = total_correct = 0
    total_grasp = total_correct_grasp = 0
    for r in results:
        ev = r.get('eval')
        if not ev:
            continue
        total_gt += ev['num_gt_instances']
        total_pred += ev['num_pred_instances']
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
        'num_pred_instances': total_pred,
        'num_matched': total_matched,
        'match_rate': round(total_matched / max(total_gt, 1), 4),
        'pred_gt_ratio': round(total_pred / max(total_gt, 1), 4),
        'redundancy_excess': total_pred - total_gt,
        'strict_e2e_all_gt': round(total_correct / max(total_gt, 1), 4),
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

    print('Loading OpenCLIP classifier...')
    ckpt = resolve_path(args.cls_checkpoint)
    thresh_path = resolve_path(args.class_thresholds)
    cls_model, preprocess, class_names, cls_meta = build_from_checkpoint(ckpt, device)
    class_thresholds = load_class_thresholds(thresh_path, class_names)

    if args.instance_source == 'gt_oracle' and not args.eval_split:
        print('Error: gt_oracle requires --eval-split', file=sys.stderr)
        sys.exit(1)

    refine_cfg = build_refine_config(args)
    meta = {
        'seg_config': args.seg_config,
        'seg_checkpoint': args.seg_checkpoint,
        'cls_checkpoint': str(ckpt),
        'class_thresholds': str(thresh_path),
        'bbox_pad': args.bbox_pad,
        'min_area': args.min_area,
        'min_area_shelf': args.min_area_shelf,
        'max_aspect_ratio': args.max_aspect_ratio,
        'nms_iou': args.nms_iou,
        'iou_match': args.iou_match,
        'iou_match_per_class': args.iou_match_per_class,
        'match_algorithm': args.match_algorithm,
        'merge_cc_iou': args.merge_cc_iou,
        'merge_cc_dist': args.merge_cc_dist,
        'merge_cc_classes': args.merge_cc_classes,
        'refine_morph_close': args.refine_morph_close,
        'refine_morph_classes': args.refine_morph_classes,
        'refine_dilate': args.refine_dilate,
        'refine_erode': args.refine_erode,
        'refine_crf': args.refine_crf,
        'refine_crf_iters': args.refine_crf_iters,
        'refine_crf_classes': args.refine_crf_classes,
        'refine_split_door_wall': args.refine_split_door_wall,
        'seg_tta': args.seg_tta,
        'seg_tta_scales': args.seg_tta_scales,
        'instance_source': args.instance_source,
        'maskrcnn_checkpoint': args.maskrcnn_checkpoint,
        'm2f_config': args.m2f_config,
        'm2f_checkpoint': args.m2f_checkpoint,
        'm2f_score_thresh': args.m2f_score_thresh,
    }

    need_segmentor = args.instance_source == 'semantic'
    segmentor = None
    if need_segmentor:
        print('Loading SegMAN...')
        segmentor = SegMANSegmentor(
            args.seg_config, args.seg_checkpoint, device=device_str)
    elif args.instance_source == 'maskrcnn':
        print(f'Instance source: maskrcnn ({args.maskrcnn_checkpoint})')
    elif args.instance_source == 'm2f':
        print(f'Instance source: m2f ({args.m2f_checkpoint})')
    elif args.instance_source == 'gt_oracle':
        print('Instance source: gt_oracle (upper bound, eval only)')

    ext_predictor = None
    if args.instance_source in ('maskrcnn', 'm2f'):
        ext_predictor = build_instance_predictor(args)

    results = []

    if args.image:
        img_path = resolve_path(args.image)
        stem = img_path.stem
        rgb = load_rgb(img_path)
        one, pred_label, pred_instances = process_one_image(
            stem, rgb, segmentor, cls_model, preprocess, class_names,
            class_thresholds, device, args, gt_label=None, refine_cfg=refine_cfg,
            instance_predictor=ext_predictor)
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
        if args.instance_source == 'gt_oracle':
            if gt_label is None:
                print(f'  skip {stem}: gt_oracle requires GT labels', file=sys.stderr)
                continue
            ip = build_instance_predictor(args, gt_label=gt_label)
        elif args.instance_source in ('maskrcnn', 'm2f'):
            ip = ext_predictor
        else:
            ip = None
        one, pred_label, pred_instances = process_one_image(
            stem, rgb, segmentor, cls_model, preprocess, class_names,
            class_thresholds, device, args, gt_label=gt_label, refine_cfg=refine_cfg,
            instance_predictor=ip)
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
