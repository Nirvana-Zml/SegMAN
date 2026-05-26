"""WiSE-FT weight interpolation for partially fine-tuned OpenCLIP visual encoder."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def clip_visual_state_dict(clip: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in clip.named_parameters()
        if name.startswith('visual.')
    }


def build_effective_finetuned_visual(
    pretrain_visual: dict[str, torch.Tensor],
    finetuned_partial: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Merge partial fine-tune weights onto pretrained visual params."""
    merged = copy.deepcopy(pretrain_visual)
    for name, tensor in finetuned_partial.items():
        if name in merged and merged[name].shape == tensor.shape:
            merged[name] = tensor.detach().cpu().clone()
    return merged


def interpolate_visual_state(
    pretrain_visual: dict[str, torch.Tensor],
    finetuned_visual: dict[str, torch.Tensor],
    alpha: float,
    keys: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """WiSE-FT: theta = alpha * finetuned + (1 - alpha) * pretrained."""
    alpha = float(alpha)
    out = copy.deepcopy(pretrain_visual)
    use_keys = keys if keys is not None else list(finetuned_visual.keys())
    for name in use_keys:
        if name not in pretrain_visual or name not in finetuned_visual:
            continue
        pre = pretrain_visual[name]
        ft = finetuned_visual[name]
        if pre.shape != ft.shape:
            continue
        out[name] = (alpha * ft + (1.0 - alpha) * pre).to(dtype=pre.dtype)
    return out


def load_visual_state(clip: nn.Module, visual_state: dict[str, torch.Tensor]) -> None:
    current = clip.state_dict()
    to_load = {}
    for name, tensor in visual_state.items():
        if name in current and current[name].shape == tensor.shape:
            to_load[name] = tensor
    missing, unexpected = clip.load_state_dict(to_load, strict=False)
    if unexpected:
        raise RuntimeError(f'unexpected keys loading visual state: {unexpected[:5]}')
