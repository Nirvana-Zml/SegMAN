#!/usr/bin/env python3
"""§8 hyperparameter sweep: run experiments sequentially and pick best."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# §8.1 priority 1–4 (one knob per experiment vs baseline lr=1e-3 linear)
EXPERIMENTS = [
    {
        'id': 's8_lr1e2_linear',
        'desc': 'P1 lr=1e-2 linear',
        'args': ['--lr', '1e-2', '--head', 'linear'],
    },
    {
        'id': 's8_lr5e4_linear',
        'desc': 'P1 lr=5e-4 linear',
        'args': ['--lr', '5e-4', '--head', 'linear'],
    },
    {
        'id': 's8_mlp5e4',
        'desc': 'P2 MLP head lr=5e-4',
        'args': [
            '--lr', '5e-4', '--head', 'mlp', '--mlp-hidden', '256',
            '--mlp-dropout', '0.1',
        ],
    },
    {
        'id': 's8_lr1e3_noweight',
        'desc': 'P3 no class weights',
        'args': ['--lr', '1e-3', '--head', 'linear', '--no-class-weights'],
    },
    {
        'id': 's8_lr1e3_ls0',
        'desc': 'P4 label_smoothing=0',
        'args': ['--lr', '1e-3', '--head', 'linear', '--label-smoothing', '0'],
    },
]

BASELINE = {
    'id': 't1_freeze_vitb16',
    'work_dir': 'outputs/openclip_classifier/t1_freeze_vitb16',
    'val_acc': 0.7018,
    'macro_f1': 0.6767,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep-root', default='outputs/openclip_classifier/sweep_s8')
    p.add_argument('--roi-root', default='data/trans10k_roi_gt')
    p.add_argument('--class-weights', default='data/trans10k_roi_gt/meta/class_weights.npy')
    p.add_argument('--ids', nargs='*', default=None, help='Run subset of experiment ids')
    p.add_argument('--skip-eval-segman', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def rel_posix(p: Path) -> str:
    return p.relative_to(PROJECT_ROOT).as_posix()


def read_best_metrics(work_dir: Path) -> dict:
    summary = work_dir / 'eval_gt_roi' / 'summary.json'
    if not summary.is_file():
        return {'val_acc': None, 'macro_f1': None}
    data = json.loads(summary.read_text(encoding='utf-8'))
    return {'val_acc': data.get('top1_acc'), 'macro_f1': data.get('macro_f1')}


def run_train(exp: dict, work_dir: Path, roi_root: str, class_weights: str | None, dry_run: bool):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'transgrasp/classification/train_openclip_classifier.py'),
        '--roi-root', roi_root,
        '--work-dir', rel_posix(work_dir),
        '--clip-model', 'ViT-B-16',
        '--clip-pretrained', 'laion2b_s34b_b88k',
        '--freeze-clip',
        '--epochs', '40',
        '--batch-size', '64',
        '--weight-decay', '0.01',
        '--label-smoothing', '0.1',
        '--num-workers', '4',
        '--patience', '8',
        '--seed', '42',
        *exp['args'],
    ]
    if class_weights and '--no-class-weights' not in exp['args']:
        cmd.extend(['--class-weights', class_weights])
    print(f'\n=== Train {exp["id"]}: {exp["desc"]} ===')
    print(' '.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def run_eval_segman(checkpoint: Path, report_dir: Path, roi_root: str, dry_run: bool):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'transgrasp/classification/eval_openclip_classifier.py'),
        '--checkpoint', rel_posix(checkpoint),
        '--roi-root', 'data/trans10k_roi_segman',
        '--split', 'val',
        '--report-dir', rel_posix(report_dir),
    ]
    print(' '.join(cmd))
    if dry_run:
        return None
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    summary = report_dir / 'summary.json'
    if summary.is_file():
        return json.loads(summary.read_text(encoding='utf-8'))
    return None


def main():
    args = parse_args()
    sweep_root = Path(args.sweep_root)
    if not sweep_root.is_absolute():
        sweep_root = PROJECT_ROOT / sweep_root
    sweep_root.mkdir(parents=True, exist_ok=True)

    exps = EXPERIMENTS
    if args.ids:
        exps = [e for e in EXPERIMENTS if e['id'] in args.ids]

    results = []
    if (sweep_root / 'sweep_results.json').is_file():
        results = json.loads((sweep_root / 'sweep_results.json').read_text(encoding='utf-8'))

    done_ids = {r['id'] for r in results}
    for exp in exps:
        if exp['id'] in done_ids:
            print(f'Skip {exp["id"]} (already in sweep_results.json)')
            continue
        work_dir = sweep_root / exp['id']
        cw = None if '--no-class-weights' in exp['args'] else args.class_weights
        run_train(exp, work_dir, args.roi_root, cw, args.dry_run)
        metrics = read_best_metrics(work_dir)
        row = {
            'id': exp['id'],
            'desc': exp['desc'],
            'work_dir': str(work_dir.relative_to(PROJECT_ROOT)),
            **metrics,
            'finished_at': datetime.now().isoformat(timespec='seconds'),
        }
        if not args.skip_eval_segman and not args.dry_run and metrics.get('val_acc') is not None:
            seg_dir = work_dir / 'eval_segman_roi'
            seg = run_eval_segman(work_dir / 'best.pth', seg_dir, args.roi_root, args.dry_run)
            if seg:
                row['segman_acc'] = seg.get('top1_acc')
                row['segman_macro_f1'] = seg.get('macro_f1')
        results.append(row)
        (sweep_root / 'sweep_results.json').write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    # rank vs baseline
    all_rows = [{'id': BASELINE['id'], 'val_acc': BASELINE['val_acc'], 'desc': 'baseline'}]
    all_rows.extend(results)
    ranked = sorted(
        [r for r in all_rows if r.get('val_acc') is not None],
        key=lambda x: x['val_acc'], reverse=True)
    summary = {
        'baseline': BASELINE,
        'ranked': ranked,
        'best': ranked[0] if ranked else None,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
    }
    (sweep_root / 'sweep_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n=== Sweep summary ===')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
