#!/usr/bin/env python3
"""For GT class bowl: histogram of predicted classes (bowl confusion diagnosis)."""
from __future__ import annotations

import argparse
import os.path as osp
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, scatter
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

CLASSES = (
    'background', 'box', 'bottle', 'window', 'eyeglass', 'freezer',
    'jar_kettle', 'door', 'cup', 'wall', 'bowl', 'shelf',
)
BOWL_ID = CLASSES.index('bowl')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--bowl-id', type=int, default=BOWL_ID)
    p.add_argument('--max-images', type=int, default=0, help='0 = all val images')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
    )

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    n_cls = len(CLASSES)
    hist = np.zeros(n_cls, dtype=np.int64)
    bowl_images = []
    n_bowl_pixels = 0
    processed = 0
    limit = args.max_images if args.max_images > 0 else len(dataset)

    for batch_indices, data in tqdm(
            zip(loader.batch_sampler, loader),
            total=min(len(loader), limit),
            desc='analyze bowl'):
        if processed >= limit:
            break
        data = scatter(data, [torch.cuda.current_device()])[0]
        with torch.no_grad():
            preds = model(return_loss=False, **data)
        for sample_idx, pred in zip(batch_indices, preds):
            if processed >= limit:
                break
            pred = np.asarray(pred, dtype=np.int64)
            ann = dataset.get_ann_info(sample_idx)
            seg_path = osp.join(dataset.ann_dir, ann['seg_map'])
            gt = mmcv.imread(seg_path, flag='unchanged', backend='pillow')
            if gt.ndim == 3:
                gt = gt[:, :, 0]
            gt = gt.astype(np.int64)
            if pred.shape != gt.shape:
                import torch.nn.functional as F
                pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
                pred = F.interpolate(
                    pred_t, size=gt.shape, mode='nearest').squeeze().numpy().astype(np.int64)

            mask = gt == args.bowl_id
            if not mask.any():
                processed += 1
                continue
            img_info = dataset.img_infos[sample_idx]
            fn = img_info.get('filename', f'idx_{sample_idx}')
            bowl_images.append(Path(fn).name)
            n_bowl_pixels += int(mask.sum())
            pred_on_bowl = pred[mask]
            for c in range(n_cls):
                hist[c] += int((pred_on_bowl == c).sum())
            processed += 1

    if n_bowl_pixels == 0:
        print('No bowl pixels in sampled val set.')
        return 1

    print(f'Bowl GT pixels: {n_bowl_pixels:,}  (images with bowl: {len(bowl_images)})')
    print(f'{"pred_class":<14} {"count":>10} {"pct":>8}')
    print('-' * 36)
    order = np.argsort(-hist)
    for c in order:
        if hist[c] == 0:
            continue
        pct = 100.0 * hist[c] / n_bowl_pixels
        mark = ' <-- correct' if c == args.bowl_id else ''
        print(f'{CLASSES[c]:<14} {hist[c]:>10,} {pct:>7.2f}%{mark}')

    correct = hist[args.bowl_id] / n_bowl_pixels
    print('-' * 36)
    print(f'bowl recall (pixel): {100.0 * correct:.2f}%')
    wrong = [(CLASSES[c], hist[c]) for c in range(n_cls) if c != args.bowl_id and hist[c] > 0]
    wrong.sort(key=lambda x: -x[1])
    if wrong:
        print('Top wrong predictions on bowl GT:')
        for name, cnt in wrong[:5]:
            print(f'  -> {name}: {100.0 * cnt / n_bowl_pixels:.2f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
