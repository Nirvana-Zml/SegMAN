#!/usr/bin/env python
"""Quick mIoU sweep over fix80k checkpoints."""
import subprocess
import sys

CKPTS = ['iter_4000', 'iter_20000', 'iter_40000', 'iter_60000', 'iter_80000']
CFG = 'local_configs/segman_trans/segman_b_trans10k_lass.py'
WD = 'outputs/trans10k_lass_mmscope_fix80k'

for name in CKPTS:
    ckpt = f'{WD}/{name}.pth'
    print(f'=== {name} ===', flush=True)
    r = subprocess.run(
        [
            sys.executable, 'tools/test.py', CFG,
            '--checkpoint', ckpt, '--eval', 'mIoU',
        ],
        capture_output=True,
        text=True,
    )
    for line in r.stdout.splitlines():
        if 'mIoU' in line and 'aAcc' in line:
            print(line)
    for line in r.stdout.splitlines()[-8:]:
        if '|' in line and '80' in line or 'mIoU' in line or 'aAcc' in line:
            pass
    # parse Summary table last numeric row
    lines = r.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == 'Summary:':
            for j in range(i + 1, min(i + 6, len(lines))):
                if lines[j].strip().startswith('|') and 'aAcc' not in lines[j]:
                    print(lines[j].strip())
                    break
    if r.returncode != 0:
        print('FAILED', r.stderr[-500:] if r.stderr else '')
