# Copyright (c) OpenMMLab. All rights reserved.
"""Penalize cup probability on GT bowl pixels (Route B balanced-v2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class BowlAntiCupLoss(nn.Module):
    """On pixels where GT == bowl, minimize P(cup) from softmax."""

    def __init__(self,
                 bowl_class_index=10,
                 cup_class_index=8,
                 loss_weight=0.25,
                 loss_name='loss_bowl_ac',
                 ignore_index=255):
        super().__init__()
        self.bowl_class_index = bowl_class_index
        self.cup_class_index = cup_class_index
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.ignore_index = ignore_index

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        del weight, avg_factor, reduction_override, kwargs
        if ignore_index == -100:
            ignore_index = self.ignore_index
        if label.dim() == 4:
            label = label.squeeze(1)
        probs = F.softmax(cls_score, dim=1)
        p_cup = probs[:, self.cup_class_index]
        mask = label == self.bowl_class_index
        if ignore_index is not None:
            mask = mask & (label != ignore_index)
        if not mask.any():
            return cls_score.sum() * 0.0
        # -log(1 - p_cup): push cup prob down on bowl GT
        loss = -torch.log((1.0 - p_cup).clamp(min=1e-6))
        return self.loss_weight * loss[mask].mean()

    @property
    def loss_name(self):
        return self._loss_name
