#!/usr/bin/env python3
"""Check E1 dual-metric gates vs baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASELINE = {
    'match_rate': 0.5932,
    'pred_gt_ratio': 1.148,
    'redundancy_excess': 458,
    'e2e_top1_on_matched': 0.8409,
    'wall_unmatched': 669,
    'door_unmatched': 359,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--eval-dir', type=str, required=True)
    p.add_argument('--baseline-report', type=str,
                   default='outputs/e2e_segment_classify/val_full/e2e_metrics_report.json')
    return p.parse_args()


def main():
    args = parse_args()
    report_path = Path(args.eval_dir) / 'e2e_metrics_report.json'
    if not report_path.is_file():
        raise FileNotFoundError(f'Missing {report_path}; run summarize_e2e_eval.py first')

    report = json.loads(report_path.read_text(encoding='utf-8'))
    il = report['instance_level']
    pc = report['per_class_gt_instance']

    match = il['match_rate']
    pred_gt = il.get('pred_gt_ratio', il['num_pred_instances'] / max(il['num_gt_instances'], 1))
    excess = il.get('redundancy_excess', il['num_pred_instances'] - il['num_gt_instances'])
    cls_acc = il['e2e_top1_on_matched']
    drop_rate = (BASELINE['redundancy_excess'] - excess) / BASELINE['redundancy_excess']

    wall_gt = pc.get('wall', {}).get('gt_instances', 1290)
    wall_m = pc.get('wall', {}).get('matched', 0)
    door_gt = pc.get('door', {}).get('gt_instances', 663)
    door_m = pc.get('door', {}).get('matched', 0)
    wall_unm = wall_gt - wall_m
    door_unm = door_gt - door_m

    gates = {
        'match_rate_ge_62': match >= 0.62,
        'redundancy_drop_ge_8pct': drop_rate >= 0.08 or pred_gt <= 1.10,
        'cls_on_matched_ge_83': cls_acc >= 0.83,
        'wall_unmatched_delta_ge_40': (BASELINE['wall_unmatched'] - wall_unm) >= 40,
        'door_unmatched_delta_ge_20': (BASELINE['door_unmatched'] - door_unm) >= 20,
    }
    e1_pass = gates['match_rate_ge_62'] and gates['redundancy_drop_ge_8pct'] and gates['cls_on_matched_ge_83']

    out = {
        'eval_dir': args.eval_dir,
        'metrics': {
            'match_rate': match,
            'pred_gt_ratio': pred_gt,
            'redundancy_excess': excess,
            'redundancy_drop_rate': round(drop_rate, 4),
            'e2e_top1_on_matched': cls_acc,
            'wall_unmatched': wall_unm,
            'door_unmatched': door_unm,
            'wall_unmatched_delta': BASELINE['wall_unmatched'] - wall_unm,
            'door_unmatched_delta': BASELINE['door_unmatched'] - door_unm,
        },
        'gates': gates,
        'E1_PASS': e1_pass,
    }
    out_path = Path(args.eval_dir) / 'e1_gate_check.json'
    out_path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    print(f'E1_PASS={e1_pass}  -> {out_path}')


if __name__ == '__main__':
    main()
