#!/usr/bin/env python3
"""Train Mask R-CNN (E1-lite) on Trans10K pseudo COCO instances."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transgrasp.pipelines.roi_postprocess import CLASSES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-ann', type=str,
                   default='segmentation/data/trans10k/coco_instances/train.json')
    p.add_argument('--val-ann', type=str,
                   default='segmentation/data/trans10k/coco_instances/val.json')
    p.add_argument('--img-root', type=str, default='segmentation/data/trans10k')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--max-samples', type=int, default=0,
                   help='Limit train images (0=all, use 1500 for fast MVP)')
    p.add_argument('--out-dir', type=str,
                   default='segmentation/outputs/maskrcnn_trans10k_pseudo')
    p.add_argument('--device', type=str, default='cuda:0')
    return p.parse_args()


def resolve(p: str) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class Trans10KCocoDataset(Dataset):
    def __init__(self, ann_path: Path, img_root: Path, split: str):
        from pycocotools import mask as mask_utils
        self.mask_utils = mask_utils
        coco = json.loads(ann_path.read_text(encoding='utf-8'))
        self.img_root = img_root
        self.split = split
        self.images = {im['id']: im for im in coco['images']}
        anns_by_img: dict[int, list] = {}
        for ann in coco['annotations']:
            anns_by_img.setdefault(ann['image_id'], []).append(ann)
        self.samples = []
        for img_id, img_info in sorted(self.images.items()):
            anns = anns_by_img.get(img_id, [])
            if not anns:
                continue
            self.samples.append((img_info, anns))

    def subset(self, max_samples: int):
        if max_samples > 0 and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_info, anns = self.samples[idx]
        img_dir = self.img_root / 'img_dir' / self.split
        img_path = img_dir / img_info['file_name']
        if not img_path.is_file():
            stem = Path(img_info['file_name']).stem
            for ext in ('.jpg', '.png'):
                alt = img_dir / f'{stem}{ext}'
                if alt.is_file():
                    img_path = alt
                    break
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        boxes, labels, masks = [], [], []
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann['category_id'])
            m = self.mask_utils.decode(ann['segmentation'])
            masks.append(np.asarray(m, dtype=np.uint8))

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': torch.as_tensor(np.stack(masks), dtype=torch.uint8),
            'image_id': torch.tensor([img_info['id']]),
        }
        img_t = TF.to_tensor(img)
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int):
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for images, targets in loader:
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n += 1
    return total_loss / max(n, 1)


def main():
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = resolve(args.img_root)

    device = torch.device(
        'cuda' if args.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    num_classes = len(CLASSES)

    train_ds = Trans10KCocoDataset(resolve(args.train_ann), img_root, 'train')
    if args.max_samples > 0:
        train_ds.subset(args.max_samples)
        print(f'Train subset: {len(train_ds)} images')
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    model = build_model(num_classes)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_loss = float('inf')
    log = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        elapsed = time.time() - t0
        row = {'epoch': epoch, 'loss': round(avg_loss, 4), 'sec': round(elapsed, 1)}
        log.append(row)
        print(f'Epoch {epoch}/{args.epochs} loss={avg_loss:.4f} ({elapsed:.1f}s)', flush=True)
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
            'num_classes': num_classes,
        }
        torch.save(ckpt, out_dir / f'epoch_{epoch}.pth')
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, out_dir / 'best.pth')

    (out_dir / 'train_log.json').write_text(
        json.dumps(log, indent=2) + '\n', encoding='utf-8')
    print(f'Done. best_loss={best_loss:.4f} -> {out_dir / "best.pth"}')


if __name__ == '__main__':
    main()
