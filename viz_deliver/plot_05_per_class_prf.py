"""Fig 5: per-class Precision / Recall / F1 — GT-ROI vs SegMAN-ROI."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import paths
from .io_utils import load_json
from .style import COLOR_GT, COLOR_SEG


METRICS = [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1')]


def _class_order(report: dict) -> list[str]:
    return sorted(report.keys(), key=lambda c: report[c].get('support', 0), reverse=True)


def run(
    out_dir: Path | None = None,
    report_gt: Path | None = None,
    report_segman: Path | None = None,
) -> Path:
    out_dir = out_dir or paths.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_json(report_gt or paths.PER_CLASS_GT)
    seg = load_json(report_segman or paths.PER_CLASS_SEGMAN)
    classes = _class_order(gt)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    x = np.arange(len(classes))
    w = 0.38

    for ax, (key, title) in zip(axes, METRICS):
        vals_gt = [gt[c][key] * 100 for c in classes]
        vals_seg = [seg.get(c, {}).get(key, 0) * 100 for c in classes]
        ax.bar(x - w / 2, vals_gt, w, label='GT-ROI', color=COLOR_GT)
        ax.bar(x + w / 2, vals_seg, w, label='SegMAN-ROI', color=COLOR_SEG)
        ax.set_ylabel(f'{title} (%)')
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [f'{c}\n(n={gt[c]["support"]})' for c in classes], fontsize=9)
    fig.suptitle('各类 Precision / Recall / F1 对比', y=1.01, fontsize=13)

    out = out_dir / '05_per_class_prf.png'
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    meta = {
        'classes': classes,
        'gt_roi': {c: gt[c] for c in classes},
        'segman_roi': {c: seg.get(c, {}) for c in classes},
    }
    (out_dir / '05_per_class_prf_meta.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Wrote', out)
    return out


if __name__ == '__main__':
    run()
