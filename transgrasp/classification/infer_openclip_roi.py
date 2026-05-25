#!/usr/bin/env python3
"""Single ROI image inference with trained classifier."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image

from transgrasp.classification.eval_openclip_classifier import build_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description='Infer fine class on one ROI crop')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--image', type=str, required=True)
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    project = PROJECT_ROOT
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = project / ckpt_path
    img_path = Path(args.image)
    if not img_path.is_absolute():
        img_path = project / img_path

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, preprocess_val, class_names, meta = build_from_checkpoint(ckpt_path, device)
    model.eval()

    image = preprocess_val(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)[0]
    k = min(args.topk, len(class_names))
    values, indices = probs.topk(k)
    topk = [
        {'class': class_names[i], 'confidence': round(float(values[j]), 4)}
        for j, i in enumerate(indices.tolist())
    ]
    result = {
        'image': str(img_path),
        'fine_class': topk[0]['class'],
        'confidence': topk[0]['confidence'],
        'topk': topk,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
