#!/usr/bin/env python3
"""Train P2 Stage-1 router: structure vs object."""
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
from transgrasp.classification.hier_dataset import HierStage1Dataset, load_stage1_groups
from transgrasp.classification.metrics import evaluate_model, save_report
from transgrasp.classification.openclip_encoder import OpenCLIPEncoder, load_openclip
from transgrasp.classification.roi_classifier import ClassificationHead, ROIClassifier


def parse_args():
    p = argparse.ArgumentParser(description='Train P2 hierarchical Stage-1 router')
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--roi-root', type=str, default='data/trans10k_roi_gt_hier')
    p.add_argument('--work-dir', type=str, default='outputs/openclip_classifier/p2_stage1_router')
    p.add_argument('--resume-encoder', type=str,
                   default='outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth')
    p.add_argument('--clip-model', type=str, default='ViT-B-16')
    p.add_argument('--clip-pretrained', type=str, default='laion2b_s34b_b88k')
    p.add_argument('--freeze-encoder', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--unfreeze-last-blocks', type=int, default=0)
    p.add_argument('--head', choices=['linear', 'mlp'], default='linear')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--label-smoothing', type=float, default=0.05)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--balance-stage1', action='store_true',
                   help='CE weight object class by train structure/object ratio')
    p.add_argument('--max-train-samples', type=int, default=-1)
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


def build_model(args, device, class_names: list[str], encoder_meta: dict | None = None):
    clip_model, preprocess_train, preprocess_val, feat_dim = load_openclip(
        args.clip_model, args.clip_pretrained, device)
    unfreeze = args.unfreeze_last_blocks
    if args.freeze_encoder:
        unfreeze = 0
    encoder = OpenCLIPEncoder(
        clip_model,
        freeze=args.freeze_encoder and unfreeze == 0,
        unfreeze_last_blocks=unfreeze if not args.freeze_encoder else unfreeze,
    )
    head = ClassificationHead(feat_dim, len(class_names), args.head)
    model = ROIClassifier(encoder, head).to(device)
    parent_blocks = int((encoder_meta or {}).get('unfreeze_last_blocks', 0))
    meta = {
        'task': 'p2_stage1_router',
        'clip_model': args.clip_model,
        'clip_pretrained': args.clip_pretrained,
        'head': args.head,
        'num_classes': len(class_names),
        'feat_dim': feat_dim,
        'class_names': class_names,
        'freeze_encoder': args.freeze_encoder,
        'unfreeze_last_blocks': unfreeze,
        'parent_encoder_blocks': parent_blocks,
        'resume_encoder': args.resume_encoder,
    }
    return model, preprocess_train, preprocess_val, meta


def gate_report(metrics: dict) -> dict:
    pc = metrics['per_class']
    structure_rec = pc.get('structure', {}).get('recall', 0.0)
    object_rec = pc.get('object', {}).get('recall', 0.0)
    acc = metrics['top1_acc']
    passed = acc >= 0.95 and structure_rec >= 0.98 and object_rec >= 0.92
    return {
        'top1_acc': acc,
        'structure_recall': structure_rec,
        'object_recall': object_rec,
        'gate_pass': passed,
        'thresholds': {'acc': 0.95, 'structure_recall': 0.98, 'object_recall': 0.92},
    }


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
    class_names = load_stage1_groups(roi_root)

    encoder_meta = {}
    enc_path = Path(args.resume_encoder)
    if not enc_path.is_absolute():
        enc_path = project / enc_path

    model, preprocess_train, preprocess_val, meta = build_model(args, device, class_names)
    if enc_path.is_file():
        encoder_meta = load_encoder_only(enc_path, model.encoder, device)
        meta['parent_encoder_blocks'] = int(encoder_meta.get('unfreeze_last_blocks', 0))
    else:
        print(f'Warning: encoder checkpoint not found: {enc_path}')

    if args.freeze_encoder:
        model.encoder.clip.eval()
        for p in model.encoder.clip.parameters():
            p.requires_grad = False

    train_ds = HierStage1Dataset(roi_root, 'train', transform=preprocess_train)
    val_ds = HierStage1Dataset(roi_root, 'val', transform=preprocess_val)
    if args.max_train_samples > 0:
        train_ds.rows = train_ds.rows[: args.max_train_samples]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.balance_stage1:
        n_struct = sum(1 for r in train_ds.rows if r['stage1_label'] == 'structure')
        n_obj = len(train_ds.rows) - n_struct
        # boost minority object class (index 1)
        w_obj = n_struct / max(n_obj, 1)
        class_weight = torch.tensor([1.0, w_obj], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=args.label_smoothing)
        print(f'Balanced CE: structure=1.0  object={w_obj:.3f}  (train {n_struct}/{n_obj})')
    param_groups = [{'params': model.head.parameters(), 'lr': args.lr}]
    clip_params = [p for p in model.encoder.clip.parameters() if p.requires_grad]
    if clip_params:
        param_groups.append({'params': clip_params, 'lr': args.lr * 0.1})
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
        gate = gate_report(val_metrics)
        elapsed = time.time() - t0
        row = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_acc': val_acc,
            'val_macro_f1': val_metrics['macro_f1'],
            'structure_recall': gate['structure_recall'],
            'object_recall': gate['object_recall'],
            'time_sec': round(elapsed, 1),
        }
        history.append(row)
        print(
            f'Epoch {epoch:03d}  loss={train_loss:.4f}  acc={val_acc:.4f}  '
            f'str_rec={gate["structure_recall"]:.4f}  obj_rec={gate["object_recall"]:.4f}  '
            f'({elapsed:.1f}s)')

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
            save_report(val_metrics, work_dir / 'eval_val', prefix='summary')
            with (work_dir / 'eval_val' / 'gate.json').open('w', encoding='utf-8') as f:
                json.dump(gate_report(val_metrics), f, indent=2)
        else:
            bad_epochs += 1

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f'Early stop at epoch {epoch} (patience={args.patience})')
            break

    if not saved_best and (work_dir / 'last.pth').is_file():
        import shutil
        shutil.copy2(work_dir / 'last.pth', work_dir / 'best.pth')

    with (work_dir / 'history.json').open('w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    best_metrics = evaluate_model(model, val_loader, device, class_names)
    gate = gate_report(best_metrics)
    print(f'Done. best acc={best_acc:.4f}  gate_pass={gate["gate_pass"]}')
    print(f'  structure_recall={gate["structure_recall"]:.4f}  object_recall={gate["object_recall"]:.4f}')


if __name__ == '__main__':
    main()
