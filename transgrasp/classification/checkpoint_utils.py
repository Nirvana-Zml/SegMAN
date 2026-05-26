"""Checkpoint save/load helpers for ROI classifier."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def encoder_trainable_state_dict(encoder: nn.Module) -> dict:
    """Save only CLIP parameters that require grad (partial fine-tune)."""
    clip = encoder.clip if hasattr(encoder, 'clip') else encoder
    return {
        name: param.detach().cpu().clone()
        for name, param in clip.named_parameters()
        if param.requires_grad
    }


def load_encoder_trainable_state(
    encoder: nn.Module,
    state: dict,
    device: torch.device,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load partial encoder weights; returns (missing, unexpected) key lists."""
    clip = encoder.clip if hasattr(encoder, 'clip') else encoder
    current = clip.state_dict()
    to_load = {}
    for name, tensor in state.items():
        if name not in current:
            continue
        if current[name].shape != tensor.shape:
            continue
        to_load[name] = tensor.to(device)
    missing, unexpected = clip.load_state_dict(to_load, strict=strict)
    return list(missing), list(unexpected)


def save_checkpoint(
    path: Path,
    head: torch.nn.Module,
    meta: dict,
    optimizer=None,
    epoch: int | None = None,
    val_acc: float | None = None,
    encoder: nn.Module | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'head': head.state_dict(),
        'meta': meta,
        'epoch': epoch,
        'val_acc': val_acc,
    }
    if encoder is not None:
        enc_state = encoder_trainable_state_dict(encoder)
        if enc_state:
            payload['encoder'] = enc_state
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    torch.save(payload, path)


def load_encoder_only(
    path: Path,
    encoder: nn.Module,
    device: torch.device,
) -> dict:
    """Load encoder weights from checkpoint without touching the head."""
    ckpt = torch.load(path, map_location=device)
    meta = ckpt.get('meta', {})
    if 'encoder' in ckpt:
        load_encoder_trainable_state(encoder, ckpt['encoder'], device, strict=False)
        print(f'Loaded encoder-only from {path} ({len(ckpt["encoder"])} tensors)')
    elif int(meta.get('unfreeze_last_blocks', 0)) > 0:
        print(
            'Warning: checkpoint has no encoder weights; '
            'CLIP visual tail uses OpenCLIP pretrained init.')
    return meta


def load_checkpoint(
    path: Path,
    head: torch.nn.Module,
    device: torch.device,
    optimizer=None,
    encoder: nn.Module | None = None,
):
    ckpt = torch.load(path, map_location=device)
    head.load_state_dict(ckpt['head'])
    if encoder is not None and 'encoder' in ckpt:
        missing, unexpected = load_encoder_trainable_state(
            encoder, ckpt['encoder'], device, strict=False)
        if missing:
            print(f'Encoder load: {len(missing)} missing keys (expected for partial ckpt)')
        if unexpected:
            print(f'Warning: encoder unexpected keys: {unexpected[:5]}...')
        print(f'Loaded encoder trainable weights ({len(ckpt["encoder"])} tensors) from {path}')
    elif encoder is not None and int(ckpt.get('meta', {}).get('unfreeze_last_blocks', 0)) > 0:
        print(
            'Warning: checkpoint has no encoder weights; '
            'CLIP visual tail uses OpenCLIP pretrained init.')
    if optimizer is not None and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except ValueError as e:
            print(f'Warning: skip optimizer state ({e}); head weights loaded only.')
    return ckpt
