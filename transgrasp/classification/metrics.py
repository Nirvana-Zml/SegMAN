"""Classification metrics and JSON reports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def compute_metrics(preds: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    assert preds.shape == labels.shape
    n = len(labels)
    acc = float((preds == labels).mean()) if n else 0.0
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1

    per_class = {}
    supports = cm.sum(axis=1)
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[name] = {
            'support': int(supports[i]),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1': round(float(f1), 4),
        }

    macro_f1 = float(np.mean([v['f1'] for v in per_class.values()])) if per_class else 0.0
    return {
        'num_samples': int(n),
        'top1_acc': round(acc, 4),
        'macro_f1': round(macro_f1, 4),
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'class_names': class_names,
    }


@torch.no_grad()
def evaluate_model(model, loader, device, class_names: list[str]) -> dict:
    model.eval()
    preds, labels = [], []
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1).cpu().numpy()
        preds.append(pred)
        labels.append(targets.numpy())
    preds_arr = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    labels_arr = np.concatenate(labels) if labels else np.array([], dtype=np.int64)
    return compute_metrics(preds_arr, labels_arr, class_names)


def save_report(metrics: dict, out_dir: Path, prefix: str = 'summary'):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    with (out_dir / f'{prefix}.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out_dir / 'per_class_report.json').open('w', encoding='utf-8') as f:
        json.dump(metrics.get('per_class', {}), f, indent=2, ensure_ascii=False)
    with (out_dir / 'confusion_matrix.json').open('w', encoding='utf-8') as f:
        json.dump({
            'class_names': metrics.get('class_names', []),
            'matrix': metrics.get('confusion_matrix', []),
        }, f, indent=2)
