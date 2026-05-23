#!/usr/bin/env python3
"""Print per-class IoU delta vs Trans10K baseline (iter_80000, mIoU 80.71)."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

BASELINE_IOU = {
    'background': 96.71,
    'box': 71.47,
    'bottle': 87.77,
    'window': 66.62,
    'eyeglass': 92.85,
    'freezer': 73.90,
    'jar_kettle': 84.04,
    'door': 75.04,
    'cup': 90.91,
    'wall': 82.72,
    'bowl': 78.91,
    'shelf': 67.61,
}
BASELINE_MIOU = 80.71
ORDER = list(BASELINE_IOU.keys())


def load_ious(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, list):
        row = data[0]
    elif 'metric' in data:
        row = data['metric']
    else:
        row = data
    out = {}
    for k, v in row.items():
        if k.startswith('IoU.'):
            out[k.split('.', 1)[1]] = float(v) * 100.0
    if 'mIoU' in row:
        out['__mIoU__'] = float(row['mIoU']) * 100.0
    return out


def resolve_eval_json(pattern: str) -> Path:
    """Accept a file path or glob (shell does not expand * inside python argv)."""
    p = Path(pattern)
    if p.is_file():
        return p
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f'No eval json matched: {pattern!r}\n'
            'Hint: run test with --work-dir <dir>, then use\n'
            '  python scripts/compare_miou_vs_baseline.py '
            '<dir>/eval_single_scale_*.json')
    if len(matches) > 1:
        print(f'Note: multiple matches, using latest: {matches[-1]}')
    return Path(matches[-1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        'eval_json',
        help='tools/test.py metric json path, or glob e.g. '
        'outputs/.../eval_final/eval_single_scale_*.json',
    )
    args = p.parse_args()
    ious = load_ious(resolve_eval_json(args.eval_json))
    m = ious.get('__mIoU__')
    if m is None:
        m = sum(ious[c] for c in ORDER) / len(ORDER)

    up = down = flat = 0
    print(f'mIoU: {m:.2f}%  (baseline {BASELINE_MIOU:.2f}%, Δ {m - BASELINE_MIOU:+.2f})')
    print(f'{"class":<12} {"IoU":>7} {"base":>7} {"Δ":>7}  trend')
    print('-' * 48)
    for c in ORDER:
        v = ious[c]
        b = BASELINE_IOU[c]
        d = v - b
        if d > 0.2:
            trend, up = '↑', up + 1
        elif d < -0.2:
            trend, down = '↓', down + 1
        else:
            trend, flat = '≈', flat + 1
        print(f'{c:<12} {v:7.2f} {b:7.2f} {d:+7.2f}  {trend}')
    print('-' * 48)
    print(f'vs baseline: ↑ {up}  ≈ {flat}  ↓ {down}  (|Δ|>0.2)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
