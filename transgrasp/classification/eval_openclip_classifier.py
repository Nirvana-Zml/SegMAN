#!/usr/bin/env python3
"""Evaluate trained OpenCLIP ROI classifier."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from transgrasp.classification.checkpoint_utils import load_checkpoint
from transgrasp.classification.dataset import ROIDataset, load_class_names
from transgrasp.classification.metrics import evaluate_model, save_report
from transgrasp.classification.openclip_encoder import OpenCLIPEncoder, load_openclip
from transgrasp.classification.roi_classifier import ClassificationHead, ROIClassifier


def parse_args():
    p = argparse.ArgumentParser(description='Eval OpenCLIP ROI classifier')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--roi-root', type=str, default='data/trans10k_roi_gt')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--report-dir', type=str, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def build_from_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    meta = ckpt['meta']
    class_names = meta['class_names']
    clip_model, _, preprocess_val, feat_dim = load_openclip(
        meta['clip_model'], meta['clip_pretrained'], device)
    encoder = OpenCLIPEncoder(
        clip_model,
        freeze=True,
        unfreeze_last_blocks=int(meta.get('unfreeze_last_blocks', 0)),
    )
    head = ClassificationHead(
        meta.get('feat_dim', feat_dim),
        meta['num_classes'],
        meta.get('head', 'linear'),
        meta.get('mlp_hidden', 256),
        meta.get('mlp_dropout', 0.1),
    )
    model = ROIClassifier(encoder, head).to(device)
    head.load_state_dict(ckpt['head'])
    return model, preprocess_val, class_names, meta


def main():
    args = parse_args()
    project = PROJECT_ROOT
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = project / ckpt_path
    roi_root = Path(args.roi_root)
    if not roi_root.is_absolute():
        roi_root = project / roi_root
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = project / report_dir

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, preprocess_val, class_names, meta = build_from_checkpoint(ckpt_path, device)

    ds = ROIDataset(roi_root, args.split, transform=preprocess_val, class_names=class_names)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    metrics = evaluate_model(model, loader, device, class_names)
    save_report(metrics, report_dir)

    out = {
        'checkpoint': str(ckpt_path),
        'roi_root': str(roi_root),
        'split': args.split,
        'top1_acc': metrics['top1_acc'],
        'macro_f1': metrics['macro_f1'],
        'num_samples': metrics['num_samples'],
    }
    with (report_dir / 'eval_meta.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(f'[{args.split}] acc={metrics["top1_acc"]:.4f}  macro_f1={metrics["macro_f1"]:.4f}')
    print(f'Report -> {report_dir}')


if __name__ == '__main__':
    main()
