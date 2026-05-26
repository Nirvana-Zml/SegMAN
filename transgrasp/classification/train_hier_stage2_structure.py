#!/usr/bin/env python3
"""Train P2 Stage-2 structure specialist: door / wall / window."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transgrasp.classification.checkpoint_utils import load_encoder_only, save_checkpoint
from transgrasp.classification.hier_dataset import HierStage2StructureDataset, load_stage2_structure
from transgrasp.classification.metrics import evaluate_model, save_report
from transgrasp.classification.openclip_encoder import OpenCLIPEncoder, load_openclip
from transgrasp.classification.roi_classifier import ClassificationHead, ROIClassifier


def parse_args():
    p = argparse.ArgumentParser(description='Train P2 Stage-2 structure head')
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--roi-root', type=str, default='data/trans10k_roi_gt_hier')
    p.add_argument('--work-dir', type=str, default='outputs/openclip_classifier/p2_stage2_structure')
    p.add_argument('--resume-encoder', type=str,
                   default='outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth')
    p.add_argument('--clip-model', type=str, default='ViT-B-16')
    p.add_argument('--clip-pretrained', type=str, default='laion2b_s34b_b88k')
    p.add_argument('--freeze-encoder', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--unfreeze-last-blocks', type=int, default=0)
    p.add_argument('--head', choices=['linear', 'mlp'], default='linear')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--patience', type=int, default=5)
    return p.parse_args()


def load_yaml_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise ImportError('PyYAML required for --config: pip install pyyaml') from e
    with path.open(encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def apply_config(args, cfg: dict):
    cli = {tok.split('=')[0].lstrip('-').replace('-', '_')
           for tok in sys.argv[1:] if tok.startswith('--')}
    for key, val in cfg.items():
        key = key.replace('-', '_')
        if hasattr(args, key) and key not in cli:
            setattr(args, key, val)
    if cfg.get('freeze_encoder') and 'freeze_encoder' not in cli:
        args.freeze_encoder = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    project = PROJECT_ROOT
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = project / cfg_path
        apply_config(args, load_yaml_config(cfg_path))

    roi_root = Path(args.roi_root)
    if not roi_root.is_absolute():
        roi_root = project / roi_root
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = project / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    class_names = load_stage2_structure(roi_root)

    enc_path = Path(args.resume_encoder)
    if not enc_path.is_absolute():
        enc_path = project / enc_path

    clip_model, preprocess_train, preprocess_val, feat_dim = load_openclip(
        args.clip_model, args.clip_pretrained, device)
    unfreeze = 0 if args.freeze_encoder else args.unfreeze_last_blocks
    encoder = OpenCLIPEncoder(
        clip_model, freeze=args.freeze_encoder, unfreeze_last_blocks=unfreeze)
    head = ClassificationHead(feat_dim, len(class_names), args.head)
    model = ROIClassifier(encoder, head).to(device)

    if enc_path.is_file():
        load_encoder_only(enc_path, model.encoder, device)
    if args.freeze_encoder:
        model.encoder.clip.eval()
        for p in model.encoder.clip.parameters():
            p.requires_grad = False

    meta = {
        'task': 'p2_stage2_structure',
        'clip_model': args.clip_model,
        'clip_pretrained': args.clip_pretrained,
        'class_names': class_names,
        'num_classes': len(class_names),
        'feat_dim': feat_dim,
        'freeze_encoder': args.freeze_encoder,
        'resume_encoder': str(enc_path),
    }

    train_ds = HierStage2StructureDataset(roi_root, 'train', transform=preprocess_train)
    val_ds = HierStage2StructureDataset(roi_root, 'val', transform=preprocess_val)
    print(f'Structure ROIs: train={len(train_ds)}  val={len(val_ds)}')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    param_groups = [{'params': model.head.parameters(), 'lr': args.lr}]
    clip_params = [p for p in model.encoder.clip.parameters() if p.requires_grad]
    if clip_params:
        param_groups.append({'params': clip_params, 'lr': args.lr * 0.2})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    with (work_dir / 'train_args.json').open('w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, default=str)

    history = []
    best_acc = -1.0
    bad_epochs = 0
    saved_best = False

    for epoch in range(args.epochs):
        model.train()
        if args.freeze_encoder:
            model.encoder.clip.eval()
        running_loss = 0.0
        n_samples = 0
        t0 = time.time()
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            n_samples += targets.size(0)
            running_loss += loss.item() * targets.size(0)

        train_loss = running_loss / max(n_samples, 1)
        val_metrics = evaluate_model(model, val_loader, device, class_names)
        val_acc = val_metrics['top1_acc']
        elapsed = time.time() - t0
        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_acc': val_acc,
            'val_macro_f1': val_metrics['macro_f1'],
            'time_sec': round(elapsed, 1),
        })
        door_f1 = val_metrics['per_class'].get('door', {}).get('f1', 0)
        wall_f1 = val_metrics['per_class'].get('wall', {}).get('f1', 0)
        print(
            f'Epoch {epoch:03d}  loss={train_loss:.4f}  acc={val_acc:.4f}  '
            f'macro_f1={val_metrics["macro_f1"]:.4f}  door_f1={door_f1:.4f}  wall_f1={wall_f1:.4f}')

        save_checkpoint(
            work_dir / 'last.pth', model.head, meta, optimizer, epoch, val_acc,
            encoder=model.encoder if not args.freeze_encoder else None)
        if val_acc >= best_acc:
            best_acc = val_acc
            bad_epochs = 0
            save_checkpoint(
                work_dir / 'best.pth', model.head, meta, optimizer, epoch, val_acc,
                encoder=model.encoder if not args.freeze_encoder else None)
            saved_best = True
            save_report(val_metrics, work_dir / 'eval_structure_val', prefix='summary')
        else:
            bad_epochs += 1
        if args.patience > 0 and bad_epochs >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break

    if not saved_best and (work_dir / 'last.pth').is_file():
        import shutil
        shutil.copy2(work_dir / 'last.pth', work_dir / 'best.pth')

    with (work_dir / 'history.json').open('w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f'Done. best structure-subset acc={best_acc:.4f}')


if __name__ == '__main__':
    main()
