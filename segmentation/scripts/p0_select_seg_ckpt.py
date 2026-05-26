#!/usr/bin/env python3
"""Pick P0 segmentation checkpoint per OpenCLIP_细分类_未达80%原因与优化方案.md §5.2 P0-1-3."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# P0-0 baseline (v2@6k eval_p0_baseline)
P0_BASELINE = {
    'mIoU': 81.80,
    'shelf': 67.73,
    'door': 73.34,
    'box': 72.97,
    'wall': 83.54,
}
WEAK_CLASSES = ('shelf', 'door', 'box')
MIOU_MIN = 81.0
WEAK_DELTA_MIN = 1.0
WALL_DROP_MAX = 1.0


def load_eval_json(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding='utf-8'))
    row = data['metric'] if 'metric' in data else data[0]['metric'] if isinstance(data, list) else data
    out = {'mIoU': float(row['mIoU']) * 100.0}
    for k, v in row.items():
        if k.startswith('IoU.'):
            out[k.split('.', 1)[1]] = float(v) * 100.0
    return out


def weak_avg(metrics: dict[str, float]) -> float:
    return sum(metrics[c] for c in WEAK_CLASSES) / len(WEAK_CLASSES)


def score_ckpt(name: str, metrics: dict[str, float]) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if metrics['mIoU'] < MIOU_MIN:
        ok = False
        notes.append(f'mIoU {metrics["mIoU"]:.2f}% < {MIOU_MIN}%')
    w_avg = weak_avg(metrics)
    w_base = weak_avg(P0_BASELINE)
    if w_avg < w_base + WEAK_DELTA_MIN:
        ok = False
        notes.append(
            f'weak avg {w_avg:.2f}% < baseline+{WEAK_DELTA_MIN} ({w_base + WEAK_DELTA_MIN:.2f}%)')
    wall_drop = P0_BASELINE['wall'] - metrics['wall']
    if wall_drop > WALL_DROP_MAX:
        ok = False
        notes.append(f'wall drop {wall_drop:.2f} pt > {WALL_DROP_MAX}')
    return ok, notes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--work-dir',
        type=str,
        default='outputs/trans10k_lass_mmscope_balanced_v2_p0weak',
        help='P0 weak finetune work dir',
    )
    p.add_argument('--write', type=str, default=None, help='Write chosen ckpt basename to this file')
    args = p.parse_args()
    work = Path(args.work_dir)
    if not work.is_absolute():
        work = Path(__file__).resolve().parents[1] / work

    candidates: list[tuple[str, dict[str, float]]] = []
    for sub in ('eval_iter_2000', 'eval_iter_4000', 'eval_best_mIoU_iter_2000'):
        matches = sorted(glob.glob(str(work / sub / 'eval_single_scale_*.json')))
        if matches:
            name = sub.replace('eval_', '')
            candidates.append((name, load_eval_json(Path(matches[-1]))))

    if not candidates:
        print('No eval JSON found. Run tools/test.py for iter_2000 / iter_4000 first.', file=sys.stderr)
        return 1

    print(f'P0-0 baseline: mIoU={P0_BASELINE["mIoU"]:.2f}%, '
          f'weak_avg={weak_avg(P0_BASELINE):.2f}%, wall={P0_BASELINE["wall"]:.2f}%')
    print('-' * 72)
    best_name = None
    best_metrics = None
    best_weak = -1.0
    for name, m in candidates:
        ok, notes = score_ckpt(name, m)
        flag = 'PASS' if ok else 'FAIL'
        print(f'{name}: mIoU={m["mIoU"]:.2f}% weak_avg={weak_avg(m):.2f}% '
              f'shelf={m["shelf"]:.2f} door={m["door"]:.2f} box={m["box"]:.2f} wall={m["wall"]:.2f} [{flag}]')
        if notes:
            print('  ', '; '.join(notes))
        if ok and weak_avg(m) > best_weak:
            best_weak = weak_avg(m)
            best_name = name
            best_metrics = m

    if best_name is None:
        # fallback: highest weak_avg among those with mIoU >= 81.0
        pool = [(n, m) for n, m in candidates if m['mIoU'] >= MIOU_MIN]
        if not pool:
            pool = candidates
        best_name, best_metrics = max(pool, key=lambda x: weak_avg(x[1]))
        print(f'\nNo checkpoint passed all rules; fallback -> {best_name} (max weak_avg @ mIoU>={MIOU_MIN})')

    ckpt_file = f'{best_name}.pth' if not best_name.startswith('best_') else f'{best_name}.pth'
    if best_name == 'iter_2000' and (work / 'best_mIoU_iter_2000.pth').is_file():
        ckpt_file = 'best_mIoU_iter_2000.pth'
    print(f'\nP0_SEG_CKPT: {ckpt_file}')
    if args.write:
        Path(args.write).write_text(ckpt_file + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
