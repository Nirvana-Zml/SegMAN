"""Linear / MLP classification head on OpenCLIP ROI features."""
from __future__ import annotations

import torch
import torch.nn as nn

from .openclip_encoder import OpenCLIPEncoder


class ClassificationHead(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        head: str = 'linear',
        mlp_hidden: int = 256,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        head = head.lower()
        if head == 'linear':
            self.net = nn.Linear(feat_dim, num_classes)
        elif head == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(feat_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_hidden, num_classes),
            )
        else:
            raise ValueError(f'Unknown head: {head}')
        self.head_type = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ROIClassifier(nn.Module):
    def __init__(
        self,
        encoder: OpenCLIPEncoder,
        head: ClassificationHead,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)
        return self.head(feats)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)
