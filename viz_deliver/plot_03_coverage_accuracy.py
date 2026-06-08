"""Fig 3: Coverage–Accuracy curves — GT-ROI vs SegMAN-ROI."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from . import paths
from .io_utils import load_json
from .style import COLOR_GT, COLOR_SEG


def _curve_points(data: dict, use_rank: bool = True) -> tuple[list[float], list[float]]:
    key = 'rank_curve' if use_rank else 'threshold_curve'
    rows = data[key]
    cov = [r['coverage'] * 100 for r in rows]
    acc = [r['accuracy_on_covered'] * 100 for r in rows]
    return cov, acc


def run(
    out_dir: Path | None = None,
    coverage_gt: Path | None = None,
    coverage_segman: Path | None = None,
) -> Path:
    out_dir = out_dir or paths.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_json(coverage_gt or paths.COVERAGE_GT)
    seg = load_json(coverage_segman or paths.COVERAGE_SEGMAN)
    cov_gt, acc_gt = _curve_points(gt)
    cov_seg, acc_seg = _curve_points(seg)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(cov_gt, acc_gt, 'o-', linewidth=2, markersize=5,
            label=f"GT-ROI (global {gt['global_top1_acc']*100:.2f}%)", color=COLOR_GT)
    ax.plot(cov_seg, acc_seg, 's-', linewidth=2, markersize=5,
            label=f"SegMAN-ROI (global {seg['global_top1_acc']*100:.2f}%)", color=COLOR_SEG)

    # Plan B gate markers
    h = gt.get('highlights', {})
    if 'acc_at_60pct_coverage' in h:
        ax.axhline(h['acc_at_60pct_coverage'] * 100, color=COLOR_GT, ls='--', alpha=0.35)
        ax.axvline(60, color='gray', ls=':', alpha=0.5)
        ax.scatter([60], [h['acc_at_60pct_coverage'] * 100], s=80, color=COLOR_GT, zorder=5)
        ax.annotate(f"@60% cov: {h['acc_at_60pct_coverage']*100:.2f}%",
                    xy=(60, h['acc_at_60pct_coverage'] * 100),
                    xytext=(62, h['acc_at_60pct_coverage'] * 100 - 4),
                    fontsize=9, color=COLOR_GT)

    ax.set_xlabel('Coverage (%)')
    ax.set_ylabel('Accuracy on covered subset (%)')
    ax.set_title('Coverage–Accuracy 曲线（rank_curve）')
    ax.set_xlim(0, 105)
    ax.set_ylim(60, 100)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)

    out = out_dir / '03_coverage_accuracy.png'
    fig.savefig(out)
    plt.close(fig)

    meta = {
        'gt_roi': {'global_top1_acc': gt['global_top1_acc'], 'highlights': gt.get('highlights')},
        'segman_roi': {'global_top1_acc': seg['global_top1_acc'], 'highlights': seg.get('highlights')},
    }
    (out_dir / '03_coverage_accuracy_meta.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Wrote', out)
    return out


if __name__ == '__main__':
    run()
