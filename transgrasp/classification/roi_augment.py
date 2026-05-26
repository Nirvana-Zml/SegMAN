"""P3 ROI augmentations: ColorJitter, light rotation, same-class CutMix."""
from __future__ import annotations

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def build_p3_pil_augment():
    return T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomRotation(degrees=5),
    ])


def apply_p3_pil_augment(pil_image, augment_fn):
    return augment_fn(pil_image)


def cutmix_same_class(
    images: torch.Tensor,
    targets: torch.Tensor,
    prob: float = 0.3,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply same-class CutMix within batch; labels unchanged."""
    if prob <= 0 or images.size(0) < 2:
        return images, targets
    if random.random() > prob:
        return images, targets

    out = images.clone()
    b, _, h, w = images.shape
    for i in range(b):
        same = [j for j in range(b) if j != i and int(targets[j]) == int(targets[i])]
        if not same:
            continue
        j = random.choice(same)
        lam = random.betavariate(alpha, alpha)
        lam = max(lam, 1.0 - lam)
        cut_w = int(w * (1.0 - lam) ** 0.5)
        cut_h = int(h * (1.0 - lam) ** 0.5)
        if cut_w <= 0 or cut_h <= 0:
            continue
        x = random.randint(0, max(w - cut_w, 0))
        y = random.randint(0, max(h - cut_h, 0))
        out[i, :, y:y + cut_h, x:x + cut_w] = images[j, :, y:y + cut_h, x:x + cut_w]
    return out, targets
