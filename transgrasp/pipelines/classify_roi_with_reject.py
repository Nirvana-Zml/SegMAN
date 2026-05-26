#!/usr/bin/env python3
"""Classify ROI folder with optional per-class rejection (Plan B4-lite)."""
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

from transgrasp.classification.dataset import ROIDataset, load_class_names
from transgrasp.classification.eval_openclip_classifier import build_from_checkpoint
from transgrasp.classification.eval_reject_policy import load_class_thresholds


def parse_args():
    p = argparse.ArgumentParser(description='ROI classify with reject')
    p.add_argument('--checkpoint', type=str,
                   default='outputs/openclip_classifier/deliver_classifier_best.pth')
    p.add_argument('--roi-root', type=str, default='data/trans10k_roi_gt')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--class-thresholds', type=str,
                   default='transgrasp/classification/configs/reject_thresholds_p3.json')
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    project = PROJECT_ROOT
    ckpt = project / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    roi_root = project / args.roi_root if not Path(args.roi_root).is_absolute() else Path(args.roi_root)
    thresh_path = project / args.class_thresholds if not Path(args.class_thresholds).is_absolute() else Path(args.class_thresholds)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project / out_path

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, preprocess_val, class_names, meta = build_from_checkpoint(ckpt, device)
    class_tau = load_class_thresholds(thresh_path, class_names)
    tau_vec = torch.tensor([class_tau[n] for n in class_names], device=device)

    ds = ROIDataset(roi_root, args.split, transform=preprocess_val, class_names=class_names)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    rows = []
    model.eval()
    for images, targets, paths in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        for i in range(len(paths)):
            pidx = int(pred[i].item())
            pname = class_names[pidx]
            c = float(conf[i].item())
            tau = class_tau[pname]
            accept = c >= tau
            rows.append({
                'path': paths[i],
                'pred_class': pname,
                'confidence': round(c, 4),
                'threshold': tau,
                'action': 'grasp' if accept else 'reject',
                'true_class': class_names[int(targets[i].item())] if targets is not None else None,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'checkpoint': str(ckpt),
        'roi_root': str(roi_root),
        'split': args.split,
        'num_samples': len(rows),
        'accept_rate': round(sum(r['action'] == 'grasp' for r in rows) / max(len(rows), 1), 4),
        'predictions': rows,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {out_path}  accept_rate={summary["accept_rate"]:.4f}')


if __name__ == '__main__':
    main()
