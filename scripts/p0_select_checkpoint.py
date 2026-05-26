#!/usr/bin/env python3
"""Pick P0 segmentation checkpoint from eval JSONs (OpenCLIP P0 §5.2 P0-1-3)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WEAK_KEYS = ('IoU.shelf', 'IoU.door', 'IoU.box')
WALL_KEY = 'IoU.wall'


def load_metric(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        data = json.load(f)
    return data['metric']


def weak_avg(metric: dict) -> float:
    return sum(metric[k] for k in WEAK_KEYS) / len(WEAK_KEYS)


def score_candidate(name: str, metric: dict, baseline: dict) -> dict:
    m = metric['mIoU'] * 100
    w = weak_avg(metric) * 100
    w0 = weak_avg(baseline) * 100
    wall = metric[WALL_KEY] * 100
    wall0 = baseline[WALL_KEY] * 100
    ok_miou = m >= 81.0
    ok_weak = w >= w0 + 1.0
    ok_wall = wall >= wall0 - 1.0
    return {
        'name': name,
        'mIoU': round(m, 2),
        'weak_avg': round(w, 2),
        'wall': round(wall, 2),
        'weak_delta': round(w - w0, 2),
        'wall_delta': round(wall - wall0, 2),
        'pass_miou': ok_miou,
        'pass_weak': ok_weak,
        'pass_wall': ok_wall,
        'pass_all': ok_miou and ok_weak and ok_wall,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-json', type=Path, required=True)
    p.add_argument('--candidate', action='append', nargs=2, metavar=('NAME', 'JSON'),
                   default=[], help='e.g. iter_2000 path/to/eval.json')
    args = p.parse_args()

    baseline = load_metric(args.baseline_json)
    rows = [score_candidate(n, load_metric(Path(j)), baseline) for n, j in args.candidate]
    passed = [r for r in rows if r['pass_all']]
    pick = passed[0]['name'] if passed else (
        max(rows, key=lambda r: (r['pass_miou'], r['weak_avg']))['name'] if rows else None
    )

    print('baseline weak_avg=%.2f%% mIoU=%.2f%% wall=%.2f%%' % (
        weak_avg(baseline) * 100, baseline['mIoU'] * 100, baseline[WALL_KEY] * 100))
    for r in rows:
        flag = 'PASS' if r['pass_all'] else 'FAIL'
        print('[%s] %s: mIoU=%.2f%% weak=%.2f%% (Δ%+.2f) wall=%.2f%% (Δ%+.2f)' % (
            flag, r['name'], r['mIoU'], r['weak_avg'], r['weak_delta'], r['wall'], r['wall_delta']))
    print('P0_SEG_CKPT suggestion:', pick or 'iter_6000 (revert)')


if __name__ == '__main__':
    main()
