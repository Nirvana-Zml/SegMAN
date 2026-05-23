# Copyright (c) OpenMMLab. All rights reserved.
"""Export HuggingFace Trans10K-v2 to MMSegmentation format.

Source repo layout (after git clone + git lfs pull):
  Trans10K-v2/
    data/train-*.parquet
    data/validation-*.parquet

Output layout (under segmentation/ by default):
  data/trans10k/
    img_dir/train/, img_dir/val/
    ann_dir/train/, ann_dir/val/

Requires: pip install datasets pyarrow
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import mmcv
import numpy as np
from PIL import Image

# RGB -> class index (see data/Trans10K-v2/README.md)
COLOR2ID = {
    (0, 0, 0): 0,  # background
    (4, 250, 7): 1,  # box
    (150, 5, 61): 2,  # bottle
    (204, 255, 4): 3,  # window
    (140, 140, 140): 4,  # eyeglass
    (6, 230, 230): 5,  # freezer
    (235, 255, 7): 6,  # jar/kettle
    (120, 120, 120): 7,  # door
    (255, 51, 7): 8,  # cup
    (224, 5, 255): 9,  # wall
    (204, 5, 255): 10,  # bowl
    (120, 120, 70): 11,  # shelf
    (255, 0, 0): 8,  # rare cup color -> cup
}

HF_SPLIT_MAP = {
    'train': 'train',
    'val': 'validation',
    'validation': 'validation',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Trans10K-v2 (HF) to MMSegmentation format')
    parser.add_argument(
        'src',
        help='Path to Trans10K-v2 repo root (contains data/*.parquet)')
    parser.add_argument(
        '-o',
        '--out-dir',
        default='data/trans10k',
        help='Output dir relative to cwd (run from segmentation/)')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='Splits to export: train val')
    parser.add_argument(
        '--mode',
        choices=['multiclass', 'binary'],
        default='multiclass',
        help='multiclass: 12 classes; binary: background vs transparent')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=-1,
        help='Debug: max samples per split (-1 = all)')
    parser.add_argument('--nproc', type=int, default=1, help='not used yet')
    return parser.parse_args()


def _check_src(src: str) -> None:
    data_dir = osp.join(src, 'data')
    if not osp.isdir(data_dir):
        raise FileNotFoundError(
            f'Missing {data_dir}. Clone HF dataset and run: cd Trans10K-v2 && git lfs pull')
    parquets = [
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and ('train' in f or 'validation' in f)
    ]
    if not parquets:
        raise FileNotFoundError(
            f'No train/validation parquet under {data_dir}. Run: git lfs pull')


def rgb_mask_to_label(mask_rgb: np.ndarray, binary: bool) -> np.ndarray:
    h, w = mask_rgb.shape[:2]
    label = np.zeros((h, w), dtype=np.uint8)
    if mask_rgb.ndim == 2:
        # already indexed
        label = mask_rgb.astype(np.uint8)
        if binary:
            label = (label > 0).astype(np.uint8)
        return label
    for color, idx in COLOR2ID.items():
        match = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        if binary and idx > 0:
            label[match] = 1
        else:
            label[match] = idx
    return label


def export_split(
    src: str,
    out_dir: str,
    split_key: str,
    binary: bool,
    max_samples: int,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError('pip install datasets pyarrow') from e

    hf_split = HF_SPLIT_MAP[split_key]
    mm_split = 'train' if split_key in ('train', 'training') else 'val'

    img_dir = osp.join(out_dir, 'img_dir', mm_split)
    ann_dir = osp.join(out_dir, 'ann_dir', mm_split)
    mmcv.mkdir_or_exist(img_dir)
    mmcv.mkdir_or_exist(ann_dir)

    ds = load_dataset(src, split=hf_split)
    n = len(ds) if max_samples < 0 else min(len(ds), max_samples)
    print(f'Export {mm_split} ({hf_split}): {n} samples -> {out_dir}')

    for i in mmcv.track_iter_progress(range(n)):
        sample = ds[i]
        img = sample['image']
        mask = sample['mask']
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        mask_rgb = np.asarray(mask.convert('RGB'))
        label = rgb_mask_to_label(mask_rgb, binary=binary)

        stem = f'{mm_split}_{i:06d}'
        img.save(osp.join(img_dir, f'{stem}.jpg'), quality=95)
        Image.fromarray(label, mode='L').save(osp.join(ann_dir, f'{stem}.png'))

    return n


def main():
    args = parse_args()
    src = osp.abspath(args.src)
    out_dir = osp.abspath(args.out_dir)
    _check_src(src)

    binary = args.mode == 'binary'
    total = 0
    for split in args.splits:
        if split not in HF_SPLIT_MAP:
            print(f'Skip unknown split: {split}', file=sys.stderr)
            continue
        total += export_split(src, out_dir, split, binary, args.max_samples)

    meta = dict(
        num_classes=2 if binary else 12,
        mode=args.mode,
        class_names=[
            'background', 'box', 'bottle', 'window', 'eyeglass', 'freezer',
            'jar_kettle', 'door', 'cup', 'wall', 'bowl', 'shelf',
        ] if not binary else ['background', 'transparent'],
    )
    mmcv.dump(meta, osp.join(out_dir, 'dataset_meta.yaml'))
    print(f'Done. {total} samples. meta -> {out_dir}/dataset_meta.yaml')
    print(f'num_classes for config: {meta["num_classes"]}')


if __name__ == '__main__':
    main()
