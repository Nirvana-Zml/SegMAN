#!/usr/bin/env python3
"""Per-class confidence rejection policy eval (Plan B3)."""
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
    p = argparse.ArgumentParser(description='Reject policy eval')
    p.add_argument('--checkpoint', type=str,
                   default='outputs/openclip_classifier/deliver_classifier_best.pth')
    p.add_argument('--roi-root', type=str, default='data/trans10k_roi_gt')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--class-thresholds', type=str,
                   default='transgrasp/classification/configs/reject_thresholds_p3.json')
    p.add_argument('--global-threshold', type=float, default=0.5)
    p.add_argument('--report-dir', type=str, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def load_class_thresholds(path: Path, class_names: list[str]) -> dict[str, float]:
    data = json.loads(path.read_text(encoding='utf-8'))
    default = float(data.get('default', 0.5))
    per = data.get('per_class', {})
    return {name: float(per.get(name, default)) for name in class_names}


@torch.no_grad()
def eval_policy(model, loader, device, class_names, class_tau, global_tau):
    model.eval()
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    tau_vec = torch.tensor(
        [class_tau[n] for n in class_names], device=device, dtype=torch.float32)

    stats = {
        'global': {'accept': 0, 'correct': 0},
        'per_class': {n: {'accept': 0, 'correct': 0, 'total': 0} for n in class_names},
    }

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        # global threshold policy
        g_accept = conf >= global_tau
        stats['global']['accept'] += int(g_accept.sum().item())
        if g_accept.any():
            stats['global']['correct'] += int((pred[g_accept] == targets[g_accept]).sum().item())

        # per-class threshold on predicted class
        pred_tau = tau_vec[pred]
        pc_accept = conf >= pred_tau
        for i in range(targets.size(0)):
            t_idx = int(targets[i].item())
            t_name = class_names[t_idx]
            stats['per_class'][t_name]['total'] += 1
            if pc_accept[i]:
                stats['per_class'][t_name]['accept'] += 1
                if int(pred[i].item()) == t_idx:
                    stats['per_class'][t_name]['correct'] += 1

    n = sum(v['total'] for v in stats['per_class'].values())
    g_cov = stats['global']['accept'] / max(n, 1)
    g_acc = stats['global']['correct'] / max(stats['global']['accept'], 1)

    pc_summary = {}
    total_accept = total_correct = 0
    for name in class_names:
        s = stats['per_class'][name]
        total_accept += s['accept']
        total_correct += s['correct']
        pc_summary[name] = {
            'total': s['total'],
            'accept': s['accept'],
            'coverage': round(s['accept'] / max(s['total'], 1), 4),
            'accuracy_on_accepted': round(s['correct'] / max(s['accept'], 1), 4),
            'threshold': class_tau[name],
        }

    pc_cov = total_accept / max(n, 1)
    pc_acc = total_correct / max(total_accept, 1)

    return {
        'num_samples': n,
        'global_threshold': global_tau,
        'global_policy': {
            'coverage': round(g_cov, 4),
            'accuracy_on_accepted': round(g_acc, 4),
        },
        'per_class_threshold_policy': {
            'coverage': round(pc_cov, 4),
            'accuracy_on_accepted': round(pc_acc, 4),
            'per_class': pc_summary,
        },
    }


def main():
    args = parse_args()
    project = PROJECT_ROOT
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = project / ckpt
    roi_root = Path(args.roi_root)
    if not roi_root.is_absolute():
        roi_root = project / roi_root
    thresh_path = Path(args.class_thresholds)
    if not thresh_path.is_absolute():
        thresh_path = project / thresh_path
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = project / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, preprocess_val, class_names, meta = build_from_checkpoint(ckpt, device)
    class_tau = load_class_thresholds(thresh_path, class_names)

    ds = ROIDataset(roi_root, args.split, transform=preprocess_val, class_names=class_names)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    result = eval_policy(
        model, loader, device, class_names, class_tau, args.global_threshold)
    result['checkpoint'] = str(ckpt)
    result['roi_root'] = str(roi_root)
    result['split'] = args.split
    result['class_thresholds_file'] = str(thresh_path)

    with (report_dir / 'reject_policy.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    gp = result['global_policy']
    pp = result['per_class_threshold_policy']
    print(
        f"global tau={args.global_threshold}: cov={gp['coverage']:.4f} "
        f"acc={gp['accuracy_on_accepted']:.4f}")
    print(
        f"per-class tau: cov={pp['coverage']:.4f} "
        f"acc={pp['accuracy_on_accepted']:.4f}")
    print(f'Report -> {report_dir / "reject_policy.json"}')


if __name__ == '__main__':
    main()
