"""Fig 2: per-class match_rate — mode A vs B grouped bar."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import paths
from .io_utils import load_json
from .style import COLOR_A, COLOR_B


def _class_order(per_class: dict) -> list[str]:
    return sorted(
        per_class.keys(),
        key=lambda c: per_class[c].get('gt_instances', 0),
        reverse=True,
    )


def run(
    out_dir: Path | None = None,
    report_a: Path | None = None,
    report_b: Path | None = None,
) -> Path:
    out_dir = out_dir or paths.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    data_a = load_json(report_a or paths.E2E_REPORT_A)
    data_b = load_json(report_b or paths.E2E_REPORT_B)
    pc_a = data_a['per_class_gt_instance']
    pc_b = data_b['per_class_gt_instance']
    classes = _class_order(pc_a)

    match_a = [pc_a[c]['match_rate'] * 100 for c in classes]
    match_b = [pc_b[c]['match_rate'] * 100 for c in classes]
    gt_n = [pc_a[c]['gt_instances'] for c in classes]

    x = np.arange(len(classes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w / 2, match_a, w, label='模式 A · semantic', color=COLOR_A)
    ax.bar(x + w / 2, match_b, w, label='模式 B · grasp', color=COLOR_B)
    ax.set_ylabel('match_rate (%)')
    ax.set_title('各类 GT 实例匹配率对比（按 GT 数量降序）')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}\n(n={n})' for c, n in zip(classes, gt_n)], fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    out = out_dir / '02_per_class_match_rate.png'
    fig.savefig(out)
    plt.close(fig)

    meta = {
        'classes': classes,
        'mode_a_match_rate': dict(zip(classes, match_a)),
        'mode_b_match_rate': dict(zip(classes, match_b)),
        'gt_instances': dict(zip(classes, gt_n)),
    }
    (out_dir / '02_per_class_match_rate_meta.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Wrote', out)
    return out


if __name__ == '__main__':
    run()
