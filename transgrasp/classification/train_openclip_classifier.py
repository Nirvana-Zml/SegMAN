#!/usr/bin/env python3
"""Train OpenCLIP ROI classifier (T1/T2)."""
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

from transgrasp.classification.checkpoint_utils import load_checkpoint, save_checkpoint
from transgrasp.classification.dataset import ROIDataset, load_class_names
from transgrasp.classification.metrics import evaluate_model, save_report
from transgrasp.classification.openclip_encoder import OpenCLIPEncoder, load_openclip
from transgrasp.classification.roi_classifier import ClassificationHead, ROIClassifier


def parse_args():
    p = argparse.ArgumentParser(description='Train OpenCLIP ROI classifier')
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--roi-root', type=str, default='data/trans10k_roi_gt')
    p.add_argument('--work-dir', type=str, default='outputs/openclip_classifier/t1_freeze_vitb16')
    p.add_argument('--clip-model', type=str, default='ViT-B-16')
    p.add_argument('--clip-pretrained', type=str, default='laion2b_s34b_b88k')
    p.add_argument('--freeze-clip', action='store_true', default=False)
    p.add_argument('--unfreeze-last-blocks', type=int, default=0)
    p.add_argument('--head', choices=['linear', 'mlp'], default='linear')
    p.add_argument('--mlp-hidden', type=int, default=256)
    p.add_argument('--mlp-dropout', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--head-lr', type=float, default=None,
                   help='LR for head when fine-tuning CLIP (default: --lr)')
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--class-weights', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--patience', type=int, default=8, help='Early stopping patience (0=off)')
    p.add_argument('--max-train-samples', type=int, default=-1, help='Debug limit')
    return p.parse_args()


def load_yaml_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise ImportError('PyYAML required for --config: pip install pyyaml') from e
    with path.open(encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _cli_overrides() -> set[str]:
    flags = set()
    for tok in sys.argv[1:]:
        if tok.startswith('--'):
            flags.add(tok.split('=')[0].lstrip('-').replace('-', '_'))
    return flags


def apply_config(args, cfg: dict):
    cli = _cli_overrides()
    for key in (
        'roi_root', 'work_dir', 'clip_model', 'clip_pretrained', 'head',
        'mlp_hidden', 'mlp_dropout', 'epochs', 'batch_size', 'lr',
        'weight_decay', 'label_smoothing', 'num_workers', 'seed',
    ):
        if key in cfg and key not in cli:
            setattr(args, key, cfg[key])
    if cfg.get('freeze_clip') and 'freeze_clip' not in cli:
        args.freeze_clip = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, device, class_names: list[str]):
    clip_model, preprocess_train, preprocess_val, feat_dim = load_openclip(
        args.clip_model, args.clip_pretrained, device)
    freeze = args.freeze_clip or args.unfreeze_last_blocks == 0
    if args.unfreeze_last_blocks > 0:
        freeze = False
    encoder = OpenCLIPEncoder(
        clip_model, freeze=freeze and args.unfreeze_last_blocks == 0,
        unfreeze_last_blocks=args.unfreeze_last_blocks)
    head = ClassificationHead(
        feat_dim, len(class_names), args.head, args.mlp_hidden, args.mlp_dropout)
    model = ROIClassifier(encoder, head).to(device)
    meta = {
        'clip_model': args.clip_model,
        'clip_pretrained': args.clip_pretrained,
        'head': args.head,
        'mlp_hidden': args.mlp_hidden,
        'mlp_dropout': args.mlp_dropout,
        'num_classes': len(class_names),
        'feat_dim': feat_dim,
        'class_names': class_names,
        'freeze_clip': args.freeze_clip,
        'unfreeze_last_blocks': args.unfreeze_last_blocks,
    }
    return model, preprocess_train, preprocess_val, meta


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
    class_names = load_class_names(roi_root)

    model, preprocess_train, preprocess_val, meta = build_model(args, device, class_names)

    train_ds = ROIDataset(roi_root, 'train', transform=preprocess_train, class_names=class_names)
    val_ds = ROIDataset(roi_root, 'val', transform=preprocess_val, class_names=class_names)
    if args.max_train_samples > 0:
        train_ds.rows = train_ds.rows[: args.max_train_samples]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    class_weight = None
    if args.class_weights:
        wpath = Path(args.class_weights)
        if not wpath.is_absolute():
            wpath = project / wpath
        class_weight = torch.tensor(np.load(wpath), dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)

    head_lr = args.head_lr if args.head_lr is not None else args.lr
    param_groups = [{'params': model.head.parameters(), 'lr': head_lr}]
    clip_params = [p for p in model.encoder.clip.parameters() if p.requires_grad]
    if clip_params:
        param_groups.append({'params': clip_params, 'lr': args.lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    start_epoch = 0
    best_acc = -1.0
    if args.resume:
        rpath = Path(args.resume)
        if not rpath.is_absolute():
            rpath = project / rpath
        ckpt = load_checkpoint(rpath, model.head, device, optimizer)
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_acc = float(ckpt.get('val_acc', -1.0))
        print(f'Resumed from {rpath} epoch={start_epoch} best_acc={best_acc:.4f}')

    with (work_dir / 'train_args.json').open('w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, default=str)

    history = []
    bad_epochs = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.freeze_clip and args.unfreeze_last_blocks == 0:
            model.encoder.clip.eval()
        running_loss = 0.0
        n_samples = 0
        t0 = time.time()
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            bs = targets.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

        train_loss = running_loss / max(n_samples, 1)
        val_metrics = evaluate_model(model, val_loader, device, class_names)
        val_acc = val_metrics['top1_acc']
        elapsed = time.time() - t0
        row = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_acc': val_acc,
            'val_macro_f1': val_metrics['macro_f1'],
            'time_sec': round(elapsed, 1),
        }
        history.append(row)
        print(
            f'Epoch {epoch:03d}  loss={train_loss:.4f}  val_acc={val_acc:.4f}  '
            f'macro_f1={val_metrics["macro_f1"]:.4f}  ({elapsed:.1f}s)')

        save_checkpoint(work_dir / 'last.pth', model.head, meta, optimizer, epoch, val_acc)
        if val_acc >= best_acc:
            best_acc = val_acc
            bad_epochs = 0
            save_checkpoint(work_dir / 'best.pth', model.head, meta, optimizer, epoch, val_acc)
            save_report(val_metrics, work_dir / 'eval_gt_roi', prefix='summary')
        else:
            bad_epochs += 1

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f'Early stop at epoch {epoch} (patience={args.patience})')
            break

    with (work_dir / 'history.json').open('w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f'Done. best val_acc={best_acc:.4f}  -> {work_dir / "best.pth"}')


if __name__ == '__main__':
    main()
