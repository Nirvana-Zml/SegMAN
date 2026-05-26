#!/usr/bin/env python3
"""Sweep WiSE-FT alpha for P3 encoder vs LAION pretrained (P4 quick validation)."""
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

from transgrasp.classification.dataset import ROIDataset, load_class_names
from transgrasp.classification.metrics import evaluate_model
from transgrasp.classification.openclip_encoder import OpenCLIPEncoder, load_openclip
from transgrasp.classification.roi_classifier import ClassificationHead, ROIClassifier
from transgrasp.classification.wise_ft import (
    build_effective_finetuned_visual,
    clip_visual_state_dict,
    interpolate_visual_state,
    load_visual_state,
)


def parse_args():
    p = argparse.ArgumentParser(description='WiSE-FT alpha sweep on P3 checkpoint')
    p.add_argument('--checkpoint', type=str,
                   default='outputs/openclip_classifier/p3_p1_hardmining/best.pth')
    p.add_argument('--roi-root', type=str, default='data/trans10k_roi_gt')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--alphas', type=str, default='0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0')
    p.add_argument('--report-dir', type=str,
                   default='outputs/openclip_classifier/p4_wise_ft_sweep')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save-best', action='store_true',
                   help='Write best alpha checkpoint to report-dir/best.pth')
    return p.parse_args()


def rp(project: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else project / path


def build_eval_model(
    ckpt_path: Path,
    device: torch.device,
    alpha: float,
    pretrain_visual: dict,
    finetuned_visual: dict,
    finetuned_keys: list[str],
    meta: dict,
):
    clip_model, _, preprocess_val, feat_dim = load_openclip(
        meta['clip_model'], meta['clip_pretrained'], device)
    interp = interpolate_visual_state(
        pretrain_visual, finetuned_visual, alpha, keys=finetuned_keys)
    load_visual_state(clip_model, interp)

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
    ckpt = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(ckpt['head'])
    return ROIClassifier(encoder, head).to(device), preprocess_val


def main():
    args = parse_args()
    project = PROJECT_ROOT
    ckpt_path = rp(project, args.checkpoint)
    roi_root = rp(project, args.roi_root)
    report_dir = rp(project, args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt['meta']
    if 'encoder' not in ckpt:
        raise SystemExit(f'Checkpoint missing encoder weights: {ckpt_path}')

    ref_clip, _, _, _ = load_openclip(meta['clip_model'], meta['clip_pretrained'], device)
    pretrain_visual = clip_visual_state_dict(ref_clip)
    finetuned_visual = build_effective_finetuned_visual(
        pretrain_visual, ckpt['encoder'])
    finetuned_keys = sorted(ckpt['encoder'].keys())

    class_names = load_class_names(roi_root)
    alphas = [float(x.strip()) for x in args.alphas.split(',') if x.strip()]

    results = []
    best_row = None
    for alpha in alphas:
        model, preprocess_val = build_eval_model(
            ckpt_path, device, alpha, pretrain_visual, finetuned_visual,
            finetuned_keys, meta)
        ds = ROIDataset(roi_root, args.split, transform=preprocess_val, class_names=class_names)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        metrics = evaluate_model(model, loader, device, class_names)
        row = {
            'alpha': alpha,
            'top1_acc': metrics['top1_acc'],
            'macro_f1': metrics['macro_f1'],
            'door_f1': metrics['per_class']['door']['f1'],
            'wall_f1': metrics['per_class']['wall']['f1'],
        }
        results.append(row)
        print(
            f'alpha={alpha:.2f}  acc={row["top1_acc"]:.4f}  macro_f1={row["macro_f1"]:.4f}  '
            f'door={row["door_f1"]:.4f}  wall={row["wall_f1"]:.4f}')
        if best_row is None or row['top1_acc'] > best_row['top1_acc']:
            best_row = row

    out = {
        'checkpoint': str(ckpt_path),
        'roi_root': str(roi_root),
        'split': args.split,
        'finetuned_keys': len(finetuned_keys),
        'baseline_p3_alpha_1': next((r for r in results if r['alpha'] == 1.0), None),
        'best': best_row,
        'pass_78': best_row['top1_acc'] >= 0.78 if best_row else False,
        'results': results,
    }
    with (report_dir / 'sweep.json').open('w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(f'\nBest alpha={best_row["alpha"]:.2f}  acc={best_row["top1_acc"]:.4f}  '
          f'pass_78={out["pass_78"]}')
    print(f'Sweep -> {report_dir / "sweep.json"}')

    if args.save_best and best_row is not None:
        from transgrasp.classification.checkpoint_utils import save_checkpoint
        best_alpha = best_row['alpha']
        model, _ = build_eval_model(
            ckpt_path, device, best_alpha, pretrain_visual, finetuned_visual,
            finetuned_keys, meta)
        best_meta = dict(meta)
        best_meta['wise_ft_alpha'] = best_alpha
        best_meta['wise_ft_source'] = str(ckpt_path)
        save_checkpoint(
            report_dir / 'best.pth', model.head, best_meta, None,
            epoch=None, val_acc=best_row['top1_acc'], encoder=model.encoder)
        print(f'Saved {report_dir / "best.pth"}')


if __name__ == '__main__':
    main()
