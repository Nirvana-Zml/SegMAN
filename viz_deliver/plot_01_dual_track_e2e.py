"""Fig 1: dual-track E2E overview — grouped bar + radar."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import paths
from .io_utils import load_json
from .style import COLOR_A, COLOR_B


METRICS = [
    ('match_rate', '实例匹配率'),
    ('cls_on_matched', '匹配后分类 Acc'),
    ('strict_e2e', '严格 E2E Acc'),
    ('wall_match', 'wall 类匹配率'),
]

RUN_A = 'f1_b1_ref'
RUN_B = 'f1_m2f_e2e'


def _pick_runs(data: dict) -> tuple[dict, dict]:
    by_name = {r['name']: r for r in data['runs'] if r.get('match_rate') is not None}
    if RUN_A not in by_name or RUN_B not in by_name:
        raise KeyError(f'Expected runs {RUN_A} and {RUN_B} in f1_plan_summary.json')
    return by_name[RUN_A], by_name[RUN_B]


def plot_bar(run_a: dict, run_b: dict, out_dir: Path) -> Path:
    labels = [m[1] for m in METRICS]
    keys = [m[0] for m in METRICS]
    vals_a = [run_a[k] * 100 for k in keys]
    vals_b = [run_b[k] * 100 for k in keys]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, vals_a, w, label='模式 A · semantic', color=COLOR_A)
    ax.bar(x + w / 2, vals_b, w, label='模式 B · grasp', color=COLOR_B)
    ax.set_ylabel('百分比 (%)')
    ax.set_title('双轨 E2E 总体指标对比')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
        ax.text(i - w / 2, va + 1, f'{va:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + w / 2, vb + 1, f'{vb:.1f}', ha='center', va='bottom', fontsize=8)

    out = out_dir / '01_dual_track_e2e_bar.png'
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_radar(run_a: dict, run_b: dict, out_dir: Path) -> Path:
    labels = [m[1] for m in METRICS]
    keys = [m[0] for m in METRICS]
    vals_a = np.array([run_a[k] * 100 for k in keys])
    vals_b = np.array([run_b[k] * 100 for k in keys])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    va = np.concatenate([vals_a, vals_a[:1]])
    vb = np.concatenate([vals_b, vals_b[:1]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, va, 'o-', linewidth=2, label='模式 A · semantic', color=COLOR_A)
    ax.fill(angles, va, alpha=0.15, color=COLOR_A)
    ax.plot(angles, vb, 'o-', linewidth=2, label='模式 B · grasp', color=COLOR_B)
    ax.fill(angles, vb, alpha=0.15, color=COLOR_B)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0, 100)
    ax.set_title('双轨 E2E 雷达图', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    ax.grid(True, alpha=0.3)

    out = out_dir / '01_dual_track_e2e_radar.png'
    fig.savefig(out)
    plt.close(fig)
    return out


def run(out_dir: Path | None = None, summary_path: Path | None = None) -> list[Path]:
    out_dir = out_dir or paths.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_json(summary_path or paths.F1_PLAN_SUMMARY)
    run_a, run_b = _pick_runs(data)
    outputs = [plot_bar(run_a, run_b, out_dir), plot_radar(run_a, run_b, out_dir)]
    meta = {
        'mode_a': {k: run_a[k] for k, _ in METRICS},
        'mode_b': {k: run_b[k] for k, _ in METRICS},
        'figures': [str(p) for p in outputs],
    }
    (out_dir / '01_dual_track_e2e_meta.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return outputs


if __name__ == '__main__':
    for p in run():
        print('Wrote', p)
