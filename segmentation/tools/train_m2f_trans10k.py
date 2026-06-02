#!/usr/bin/env python3
"""Train Mask2Former on Trans10K pseudo COCO instances (F1-3)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CFG = (
    'segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=DEFAULT_CFG)
    p.add_argument('--work-dir', type=str, default='')
    p.add_argument('--max-iters', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=0)
    p.add_argument('--resume', action='store_true',
                   help='Resume from work_dir/last_checkpoint (clears load_from)')
    p.add_argument('--resume-from', type=str, default='',
                   help='Resume from explicit local .pth (recommended)')
    p.add_argument('--no-pretrain', action='store_true',
                   help='Skip COCO pretrained init (smoke / offline)')
    return p.parse_args()


def resolve(p: str) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ImportError as exc:
        raise SystemExit(
            'mmdet/mmengine not found. Run: bash scripts/setup_f1_mmdet_env.sh'
        ) from exc

    cfg_path = resolve(args.config)
    cfg = Config.fromfile(str(cfg_path))
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.max_iters > 0:
        cfg.train_cfg.max_iters = args.max_iters
        cfg.default_hooks.checkpoint.interval = min(5000, args.max_iters)
    if args.batch_size > 0:
        cfg.train_dataloader.batch_size = args.batch_size

    work_dir = resolve(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from:
        ckpt = resolve(args.resume_from)
        if not ckpt.is_file():
            raise SystemExit(f'Checkpoint not found: {ckpt}')
        cfg.load_from = None
        cfg.resume = True
        (work_dir / 'last_checkpoint').write_text(str(ckpt) + '\n', encoding='utf-8')
        print(f'Resume from {ckpt} (load_from cleared)', flush=True)
    elif args.resume:
        cfg.load_from = None
        cfg.resume = True
        print('Resume from work_dir/last_checkpoint (load_from cleared)', flush=True)
    elif args.no_pretrain:
        cfg.load_from = None

    cfg.dump(str(work_dir / 'config_resolved.py'))

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
