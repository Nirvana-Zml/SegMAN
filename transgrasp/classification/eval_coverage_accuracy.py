#!/usr/bin/env python3
"""Coverage-accuracy curve for ROI classifier (Plan B2)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transgrasp.classification.dataset import ROIDataset, load_class_names
from transgrasp.classification.eval_openclip_classifier import build_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description='Coverage-accuracy curve eval')
    p.add_argument('--checkpoint', type=str,
                   default='outputs/openclip_classifier/deliver_classifier_best.pth')
    p.add_argument('--roi-root', type=str, default='data/trans10k_roi_gt')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--report-dir', type=str, required=True)
    p.add_argument('--thresholds', type=str,
                   default='0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    confs, correct = [], []
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        confs.append(conf.cpu().numpy())
        correct.append((pred == targets).cpu().numpy())
    return np.concatenate(confs), np.concatenate(correct).astype(np.float64)


def curve_from_scores(confs: np.ndarray, correct: np.ndarray, thresholds: list[float]):
    n = len(confs)
    order = np.argsort(-confs)
    confs_s = confs[order]
    correct_s = correct[order]
    cum_correct = np.cumsum(correct_s)
    cum_n = np.arange(1, n + 1)

    rows = []
    for tau in thresholds:
        mask = confs >= tau
        k = int(mask.sum())
        if k == 0:
            rows.append({
                'threshold': tau,
                'coverage': 0.0,
                'num_covered': 0,
                'accuracy_on_covered': 0.0,
            })
            continue
        rows.append({
            'threshold': tau,
            'coverage': round(k / n, 4),
            'num_covered': k,
            'accuracy_on_covered': round(float(correct[mask].mean()), 4),
        })

    # rank-based curve (select top-k by confidence)
    rank_rows = []
    for pct in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        k = max(1, int(round(n * pct)))
        rank_rows.append({
            'coverage': round(k / n, 4),
            'num_covered': k,
            'min_conf_in_set': round(float(confs_s[k - 1]), 4),
            'accuracy_on_covered': round(float(cum_correct[k - 1] / k), 4),
        })

    global_acc = round(float(correct.mean()), 4)
    return rows, rank_rows, global_acc, n


def main():
    args = parse_args()
    project = PROJECT_ROOT
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = project / ckpt
    roi_root = Path(args.roi_root)
    if not roi_root.is_absolute():
        roi_root = project / roi_root
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = project / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, preprocess_val, class_names, meta = build_from_checkpoint(ckpt, device)
    ds = ROIDataset(roi_root, args.split, transform=preprocess_val, class_names=class_names)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    confs, correct = collect_predictions(model, loader, device)
    thresholds = [float(x.strip()) for x in args.thresholds.split(',') if x.strip()]
    threshold_curve, rank_curve, global_acc, n = curve_from_scores(confs, correct, thresholds)

    gate_60 = next((r for r in rank_curve if r['coverage'] >= 0.599), rank_curve[-1])
    gate_70 = next((r for r in rank_curve if r['coverage'] >= 0.699), rank_curve[-1])

    out = {
        'checkpoint': str(ckpt),
        'roi_root': str(roi_root),
        'split': args.split,
        'num_samples': n,
        'global_top1_acc': global_acc,
        'threshold_curve': threshold_curve,
        'rank_curve': rank_curve,
        'highlights': {
            'acc_at_60pct_coverage': gate_60['accuracy_on_covered'],
            'acc_at_70pct_coverage': gate_70['accuracy_on_covered'],
            'coverage_at_tau_0.5': next(
                (r['coverage'] for r in threshold_curve if r['threshold'] == 0.5), None),
            'acc_at_tau_0.5': next(
                (r['accuracy_on_covered'] for r in threshold_curve if r['threshold'] == 0.5), None),
        },
        'plan_b_gates': {
            'pass_acc_78_at_coverage_60': gate_60['accuracy_on_covered'] >= 0.78,
            'pass_acc_80_at_coverage_70': gate_70['accuracy_on_covered'] >= 0.80,
        },
    }
    with (report_dir / 'coverage_accuracy.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(f'[{args.split}] global_acc={global_acc:.4f}  n={n}')
    print('threshold curve (sample):')
    for r in threshold_curve:
        if r['threshold'] in (0.5, 0.6, 0.7):
            print(
                f"  tau={r['threshold']:.2f}  cov={r['coverage']:.4f}  "
                f"acc={r['accuracy_on_covered']:.4f}")
    print(f"  @60% coverage: acc={gate_60['accuracy_on_covered']:.4f}")
    print(f"  @70% coverage: acc={gate_70['accuracy_on_covered']:.4f}")
    print(f"Report -> {report_dir / 'coverage_accuracy.json'}")


if __name__ == '__main__':
    main()
