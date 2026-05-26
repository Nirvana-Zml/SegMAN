#!/usr/bin/env python3
"""Evaluate P2 hierarchical cascade on full 11-class ROI task."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transgrasp.classification.checkpoint_utils import load_encoder_only
from transgrasp.classification.dataset import ROIDataset, load_class_names
from transgrasp.classification.eval_openclip_classifier import build_from_checkpoint
from transgrasp.classification.hier_dataset import load_stage1_groups, load_stage2_structure
from transgrasp.classification.metrics import evaluate_model, save_report
from transgrasp.classification.openclip_encoder import OpenCLIPEncoder, load_openclip
from transgrasp.classification.roi_classifier import ClassificationHead, ROIClassifier


def parse_args():
    p = argparse.ArgumentParser(description='Eval P2 hierarchical classifier')
    p.add_argument('--stage1', type=str, required=True, help='Stage-1 router best.pth')
    p.add_argument('--stage2-structure', type=str, required=True, help='Stage-2 structure best.pth')
    p.add_argument('--object-head', type=str, required=True, help='P1 11-class best.pth')
    p.add_argument('--roi-root', type=str, default='data/trans10k_roi_gt')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--report-dir', type=str, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def load_router(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    meta = ckpt['meta']
    stage1_names = meta['class_names']
    clip_model, _, preprocess_val, feat_dim = load_openclip(
        meta['clip_model'], meta['clip_pretrained'], device)
    encoder = OpenCLIPEncoder(clip_model, freeze=True, unfreeze_last_blocks=0)
    head = ClassificationHead(feat_dim, len(stage1_names), meta.get('head', 'linear'))
    model = ROIClassifier(encoder, head).to(device)
    head.load_state_dict(ckpt['head'])
    enc_src = meta.get('resume_encoder')
    if enc_src:
        enc_path = Path(enc_src)
        if not enc_path.is_absolute():
            enc_path = PROJECT_ROOT / enc_path
        if enc_path.is_file():
            load_encoder_only(enc_path, model.encoder, device)
    model.eval()
    return model, preprocess_val, stage1_names


def load_structure_head(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    meta = ckpt['meta']
    names = meta['class_names']
    clip_model, _, _, feat_dim = load_openclip(
        meta['clip_model'], meta['clip_pretrained'], device)
    encoder = OpenCLIPEncoder(clip_model, freeze=True, unfreeze_last_blocks=0)
    head = ClassificationHead(feat_dim, len(names), meta.get('head', 'linear'))
    model = ROIClassifier(encoder, head).to(device)
    head.load_state_dict(ckpt['head'])
    enc_src = meta.get('resume_encoder')
    if enc_src:
        enc_path = Path(enc_src)
        if not enc_path.is_absolute():
            enc_path = PROJECT_ROOT / enc_path
        if enc_path.is_file():
            load_encoder_only(enc_path, model.encoder, device)
    model.eval()
    return model, names


class HierarchicalCascade(torch.nn.Module):
    def __init__(
        self,
        stage1: ROIClassifier,
        stage2: ROIClassifier,
        object_model: ROIClassifier,
        stage1_names: list[str],
        stage2_names: list[str],
        class_names_11: list[str],
    ):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.object_model = object_model
        self.stage1_names = stage1_names
        self.stage2_names = stage2_names
        self.class_names_11 = class_names_11
        self.structure_idx = stage1_names.index('structure')
        self.door_idx = class_names_11.index('door')
        self.wall_idx = class_names_11.index('wall')
        self.window_idx = class_names_11.index('window')
        self.stage2_to_11 = {n: class_names_11.index(n) for n in stage2_names}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.size(0)
        out = torch.full((b, len(self.class_names_11)), -1e9, device=images.device, dtype=torch.float32)
        s1 = self.stage1(images).argmax(dim=1)
        s2_logits = self.stage2(images)
        obj_logits = self.object_model(images)
        obj_logits = obj_logits.clone()
        obj_logits[:, self.door_idx] = -1e9
        obj_logits[:, self.wall_idx] = -1e9
        obj_logits[:, self.window_idx] = -1e9

        for i in range(b):
            if int(s1[i]) == self.structure_idx:
                pred2 = s2_logits[i].argmax().item()
                cls = self.stage2_names[pred2]
                out[i, self.stage2_to_11[cls]] = s2_logits[i, pred2]
            else:
                out[i] = obj_logits[i]
        return out


@torch.no_grad()
def evaluate_cascade(model, loader, device, class_names: list[str]) -> dict:
    model.eval()
    import numpy as np
    preds, labels = [], []
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(targets.numpy())
    preds_arr = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    labels_arr = np.concatenate(labels) if labels else np.array([], dtype=np.int64)
    from transgrasp.classification.metrics import compute_metrics
    return compute_metrics(preds_arr, labels_arr, class_names)


def main():
    args = parse_args()
    project = PROJECT_ROOT
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    def rp(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else project / path

    roi_root = rp(args.roi_root)
    report_dir = rp(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    stage1_model, preprocess_val, stage1_names = load_router(rp(args.stage1), device)
    stage2_model, stage2_names = load_structure_head(rp(args.stage2_structure), device)
    object_model, _, _, _ = build_from_checkpoint(rp(args.object_head), device)

    class_names = load_class_names(roi_root)
    cascade = HierarchicalCascade(
        stage1_model, stage2_model, object_model,
        stage1_names, stage2_names, class_names,
    ).to(device)

    ds = ROIDataset(roi_root, args.split, transform=preprocess_val, class_names=class_names)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    metrics = evaluate_cascade(cascade, loader, device, class_names)
    save_report(metrics, report_dir)

    out = {
        'stage1': str(rp(args.stage1)),
        'stage2_structure': str(rp(args.stage2_structure)),
        'object_head': str(rp(args.object_head)),
        'roi_root': str(roi_root),
        'split': args.split,
        'top1_acc': metrics['top1_acc'],
        'macro_f1': metrics['macro_f1'],
    }
    with (report_dir / 'eval_meta.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(f'[{args.split}] hierarchical acc={metrics["top1_acc"]:.4f}  macro_f1={metrics["macro_f1"]:.4f}')
    print(f'Report -> {report_dir}')


if __name__ == '__main__':
    main()
