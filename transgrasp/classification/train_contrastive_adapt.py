#!/usr/bin/env python3
"""Trans10K ROI-image vs class-text contrastive adapt (P4 small / P4-full)."""
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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transgrasp.classification.checkpoint_utils import load_checkpoint, save_checkpoint
from transgrasp.classification.dataset import ROIDataset, load_class_names
from transgrasp.classification.metrics import evaluate_model, save_report
from transgrasp.classification.openclip_encoder import OpenCLIPEncoder, load_openclip
from transgrasp.classification.roi_classifier import ClassificationHead, ROIClassifier


def parse_args():
    p = argparse.ArgumentParser(description='P4 contrastive domain adapt')
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--resume', type=str,
                   default='outputs/openclip_classifier/p3_p1_hardmining/best.pth')
    p.add_argument('--roi-root', type=str, default='data/trans10k_roi_gt')
    p.add_argument('--work-dir', type=str,
                   default='outputs/openclip_classifier/p4_contrastive_small')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--patience', type=int, default=0,
                   help='Contrastive early-stop patience (0=off; P4-full uses 1)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-7)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--unfreeze-last-blocks', type=int, default=4)
    p.add_argument('--max-train-samples', type=int, default=8000,
                   help='Cap train ROIs (-1 = full train split)')
    p.add_argument('--text-template', type=str, default='a transparent {name} in an indoor scene')
    p.add_argument('--head-finetune-epochs', type=int, default=2,
                   help='Short CE head tune after contrastive (0=skip)')
    p.add_argument('--no-head-finetune', action='store_true')
    p.add_argument('--head-finetune-min-delta', type=float, default=0.0,
                   help='Only head-ft if contrastive best beats resume val_acc by this margin')
    p.add_argument('--head-lr', type=float, default=1e-5)
    p.add_argument('--eval-every', type=int, default=1,
                   help='Save eval report every N contrastive epochs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
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
        'resume', 'roi_root', 'work_dir', 'epochs', 'patience', 'batch_size', 'lr',
        'weight_decay', 'unfreeze_last_blocks', 'max_train_samples', 'text_template',
        'head_finetune_epochs', 'head_finetune_min_delta', 'head_lr', 'eval_every', 'seed',
    ):
        if key in cfg and key not in cli:
            setattr(args, key, cfg[key])
    if cfg.get('no_head_finetune') and 'no_head_finetune' not in cli:
        args.no_head_finetune = True


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_prompts(class_names: list[str], template: str) -> list[str]:
    return [template.format(name=n.replace('_', ' ')) for n in class_names]


def clip_contrastive_loss(
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    image_feats = F.normalize(image_feats, dim=-1)
    text_feats = F.normalize(text_feats, dim=-1)
    logit_scale = logit_scale.exp()
    logits = logit_scale * image_feats @ text_feats.T
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) * 0.5


def build_from_resume(args, project: Path, device: torch.device):
    ckpt_path = project / args.resume if not Path(args.resume).is_absolute() else Path(args.resume)
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt['meta']
    clip_model, preprocess_train, preprocess_val, feat_dim = load_openclip(
        meta['clip_model'], meta['clip_pretrained'], device)
    encoder = OpenCLIPEncoder(
        clip_model, freeze=False, unfreeze_last_blocks=args.unfreeze_last_blocks)
    head = ClassificationHead(
        meta.get('feat_dim', feat_dim),
        meta['num_classes'],
        meta.get('head', 'linear'),
        meta.get('mlp_hidden', 256),
        meta.get('mlp_dropout', 0.1),
    )
    model = ROIClassifier(encoder, head).to(device)
    load_checkpoint(ckpt_path, model.head, device, encoder=model.encoder)
    resume_val = float(ckpt.get('val_acc', -1.0))
    return model, preprocess_train, preprocess_val, meta, ckpt_path, resume_val


def make_out_meta(meta: dict, resume_path: Path, args, method: str) -> dict:
    out = dict(meta)
    out['p4_method'] = method
    out['resume'] = str(resume_path)
    out['text_template'] = args.text_template
    out['max_train_samples'] = args.max_train_samples
    out['patience'] = args.patience
    out['contrastive_lr'] = args.lr
    return out


def main():
    args = parse_args()
    project = PROJECT_ROOT
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = project / cfg_path
        apply_config(args, load_yaml_config(cfg_path))

    if args.no_head_finetune:
        args.head_finetune_epochs = 0

    work_dir = project / args.work_dir if not Path(args.work_dir).is_absolute() else Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    roi_root = project / args.roi_root if not Path(args.roi_root).is_absolute() else Path(args.roi_root)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    class_names = load_class_names(roi_root)
    model, preprocess_train, preprocess_val, meta, resume_path, resume_val = build_from_resume(
        args, project, device)

    with (work_dir / 'train_args.json').open('w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, default=str)

    import open_clip
    tokenizer = open_clip.get_tokenizer(meta['clip_model'])
    prompts = class_prompts(class_names, args.text_template)

    clip = model.encoder.clip
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        class_text_feats = F.normalize(clip.encode_text(tokens), dim=-1)

    for p in clip.parameters():
        p.requires_grad = False
    visual = clip.visual
    blocks = getattr(getattr(visual, 'transformer', None), 'resblocks', None)
    if blocks is None and hasattr(visual, 'blocks'):
        blocks = visual.blocks
    if blocks is not None:
        n = min(args.unfreeze_last_blocks, len(blocks))
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

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

    method = 'contrastive_full' if args.max_train_samples <= 0 else 'contrastive_small'
    print(
        f'P4 contrastive: train={len(train_ds)} val={len(val_ds)} '
        f'resume_val={resume_val:.4f} patience={args.patience} lr={args.lr}')

    clip_params = [p for p in clip.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(clip_params, lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_acc = -1.0
    best_epoch = -1
    bad_epochs = 0
    contrastive_best_acc = -1.0

    for epoch in range(args.epochs):
        model.encoder.clip.train()
        running = 0.0
        n = 0
        t0 = time.time()
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            text_feats = class_text_feats[targets]
            image_feats = clip.encode_image(images)
            loss = clip_contrastive_loss(image_feats, text_feats, clip.logit_scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bs = targets.size(0)
            running += loss.item() * bs
            n += bs

        model.eval()
        val_metrics = evaluate_model(model, val_loader, device, class_names)
        val_acc = val_metrics['top1_acc']
        row = {
            'epoch': epoch,
            'train_loss': round(running / max(n, 1), 4),
            'val_acc': val_acc,
            'val_macro_f1': val_metrics['macro_f1'],
            'time_sec': round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f'[contrastive] ep={epoch} loss={row["train_loss"]:.4f} '
            f'val_acc={val_acc:.4f} macro_f1={val_metrics["macro_f1"]:.4f} '
            f'({row["time_sec"]}s)')

        out_meta = make_out_meta(meta, resume_path, args, method)
        save_checkpoint(
            work_dir / 'last.pth', model.head, out_meta, optimizer, epoch, val_acc,
            encoder=model.encoder)

        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            save_report(val_metrics, work_dir / f'eval_gt_roi_ep{epoch:02d}')

        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            contrastive_best_acc = val_acc
            save_checkpoint(
                work_dir / 'best.pth', model.head, out_meta, optimizer, epoch, val_acc,
                encoder=model.encoder)
            save_report(val_metrics, work_dir / 'eval_gt_roi_after_contrastive')
        else:
            bad_epochs += 1

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f'Contrastive early stop at epoch {epoch} (patience={args.patience})')
            break

    with (work_dir / 'contrastive_history.json').open('w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    run_head_ft = (
        args.head_finetune_epochs > 0
        and contrastive_best_acc >= resume_val + args.head_finetune_min_delta
    )
    if args.head_finetune_epochs > 0 and not run_head_ft:
        print(
            f'Skip head fine-tune: contrastive best {contrastive_best_acc:.4f} '
            f'< resume {resume_val:.4f} + {args.head_finetune_min_delta}')

    if run_head_ft:
        print(f'=== head fine-tune {args.head_finetune_epochs} epochs ===')
        for p in clip.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
        head_opt = torch.optim.AdamW(
            model.head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        hf_history = []
        head_best = best_acc
        for epoch in range(args.head_finetune_epochs):
            model.train()
            model.encoder.clip.eval()
            running = 0.0
            n = 0
            for images, targets, _ in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                head_opt.zero_grad(set_to_none=True)
                loss = criterion(model(images), targets)
                loss.backward()
                head_opt.step()
                bs = targets.size(0)
                running += loss.item() * bs
                n += bs
            val_metrics = evaluate_model(model, val_loader, device, class_names)
            val_acc = val_metrics['top1_acc']
            hf_history.append({
                'epoch': epoch,
                'train_loss': round(running / max(n, 1), 4),
                'val_acc': val_acc,
                'val_macro_f1': val_metrics['macro_f1'],
            })
            print(f'[head-ft] ep={epoch} val_acc={val_acc:.4f}')
            out_meta = make_out_meta(meta, resume_path, args, f'{method}+head_ft')
            save_checkpoint(
                work_dir / 'last.pth', model.head, out_meta, head_opt, epoch, val_acc,
                encoder=model.encoder)
            if val_acc >= head_best:
                head_best = val_acc
                best_acc = val_acc
                save_checkpoint(
                    work_dir / 'best.pth', model.head, out_meta, head_opt, epoch, val_acc,
                    encoder=model.encoder)
                save_report(val_metrics, work_dir / 'eval_gt_roi')

        with (work_dir / 'head_ft_history.json').open('w', encoding='utf-8') as f:
            json.dump(hf_history, f, indent=2)

    summary = {
        'best_epoch': best_epoch,
        'best_val_acc': best_acc,
        'contrastive_best_acc': contrastive_best_acc,
        'resume_val_acc': resume_val,
        'train_samples': len(train_ds),
    }
    with (work_dir / 'train_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'Done. best val_acc={best_acc:.4f} (ep={best_epoch}) -> {work_dir / "best.pth"}')


if __name__ == '__main__':
    main()
