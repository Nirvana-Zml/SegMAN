#!/usr/bin/env python3
"""Generate all deliver visualization figures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python viz_deliver/run_all.py` from SegMAN root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from viz_deliver import paths  # noqa: E402
from viz_deliver import style  # noqa: F401,E402 — apply rcParams
from viz_deliver.plot_01_dual_track_e2e import run as run_01
from viz_deliver.plot_02_per_class_match import run as run_02
from viz_deliver.plot_03_coverage_accuracy import run as run_03
from viz_deliver.plot_04_confusion_matrix import run as run_04
from viz_deliver.plot_05_per_class_prf import run as run_05


def main():
    parser = argparse.ArgumentParser(description='TransGrasp deliver figures')
    parser.add_argument(
        '--out-dir', type=Path, default=paths.OUT_DIR,
        help='Output directory for PNG/JSON (default: viz_deliver/output)',
    )
    parser.add_argument(
        '--only', type=int, nargs='*', choices=[1, 2, 3, 4, 5],
        help='Run selected figures only, e.g. --only 1 3',
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tasks = {
        1: ('双轨 E2E 总体对比', lambda: run_01(args.out_dir)),
        2: ('各类 match_rate', lambda: run_02(args.out_dir)),
        3: ('Coverage–Accuracy', lambda: run_03(args.out_dir)),
        4: ('混淆矩阵', lambda: run_04(args.out_dir)),
        5: ('Per-class P/R/F1', lambda: run_05(args.out_dir)),
    }
    selected = args.only or list(tasks.keys())

    written: list[Path] = []
    for idx in selected:
        label, fn = tasks[idx]
        print(f'[{idx}/5] {label} ...')
        result = fn()
        if isinstance(result, list):
            written.extend(result)
        else:
            written.append(result)

    print(f'\nDone. {len(written)} figure(s) -> {args.out_dir.resolve()}')


if __name__ == '__main__':
    main()
