"""Frozen / partial-finetune OpenCLIP image encoder wrapper."""
from __future__ import annotations

import open_clip
import torch
import torch.nn as nn


def load_openclip(
    clip_model: str,
    clip_pretrained: str,
    device: str | torch.device = 'cuda',
):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained)
    model = model.to(device)
    feat_dim = model.visual.output_dim if hasattr(model.visual, 'output_dim') else None
    if feat_dim is None:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            feat_dim = model.encode_image(dummy).shape[-1]
    return model, preprocess_train, preprocess_val, int(feat_dim)


def set_clip_trainable(model: nn.Module, freeze: bool, unfreeze_last_blocks: int = 0):
    if freeze and unfreeze_last_blocks <= 0:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return

    for p in model.parameters():
        p.requires_grad = False

    visual = model.visual
    blocks = getattr(visual, 'transformer', None)
    if blocks is not None and hasattr(blocks, 'resblocks'):
        resblocks = blocks.resblocks
        n = min(unfreeze_last_blocks, len(resblocks))
        for block in resblocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
    elif hasattr(visual, 'blocks'):
        n = min(unfreeze_last_blocks, len(visual.blocks))
        for block in visual.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    trainable = any(p.requires_grad for p in model.parameters())
    if trainable:
        model.train()
    else:
        model.eval()


class OpenCLIPEncoder(nn.Module):
    def __init__(self, model: nn.Module, freeze: bool = True, unfreeze_last_blocks: int = 0):
        super().__init__()
        self.clip = model
        set_clip_trainable(self.clip, freeze, unfreeze_last_blocks)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        trainable = any(p.requires_grad for p in self.clip.parameters())
        if not trainable:
            with torch.no_grad():
                return self.clip.encode_image(images)
        return self.clip.encode_image(images)
