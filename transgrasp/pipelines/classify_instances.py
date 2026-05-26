"""Classify a list of ROI crops with optional per-class rejection."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from transgrasp.pipelines.roi_extract import InstanceROI


@torch.no_grad()
def classify_instances(
    model,
    preprocess,
    class_names: list[str],
    class_thresholds: dict[str, float],
    instances: list[InstanceROI],
    device: torch.device,
) -> list[dict[str, Any]]:
    if not instances:
        return []

    model.eval()
    rows = []
    for inst in instances:
        pil = Image.fromarray(inst.crop_rgb)
        tensor = preprocess(pil).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        conf, pred_idx = probs.max(dim=0)
        pred_idx = int(pred_idx.item())
        pred_name = class_names[pred_idx]
        confidence = float(conf.item())
        tau = class_thresholds[pred_name]
        action = 'grasp' if confidence >= tau else 'reject'
        k = min(3, len(class_names))
        vals, idxs = probs.topk(k)
        topk = [
            {
                'class': class_names[int(idxs[j])],
                'confidence': round(float(vals[j].item()), 4),
            }
            for j in range(k)
        ]

        row = inst.to_dict()
        row.update({
            'pred_class': pred_name,
            'pred_class_id': pred_idx,
            'confidence': round(confidence, 4),
            'threshold': tau,
            'action': action,
            'topk': topk,
        })
        rows.append(row)
    return rows
