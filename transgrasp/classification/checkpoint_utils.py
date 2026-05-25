"""Checkpoint save/load helpers for ROI classifier."""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: Path,
    head: torch.nn.Module,
    meta: dict,
    optimizer=None,
    epoch: int | None = None,
    val_acc: float | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'head': head.state_dict(),
        'meta': meta,
        'epoch': epoch,
        'val_acc': val_acc,
    }
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: Path, head: torch.nn.Module, device: torch.device, optimizer=None):
    ckpt = torch.load(path, map_location=device)
    head.load_state_dict(ckpt['head'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt
