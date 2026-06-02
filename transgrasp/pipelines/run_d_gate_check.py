#!/usr/bin/env python3
"""Check scheme D gates and write d_plan_summary.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASELINE_MATCH = 0.5916
C_MATCH = 0.5862

D_GATES = {
    'match_rate_ge_61': 0.61,
    'pred_gt_ratio_le_108': 1.08,
    'strict_e2e_ge_515': 0.515,
    'cls_on_matched_ge_83': 0.83,
}

D_FAIL_MATCH = 0.60
D_FAIL_PRED_GT = 1.10

RUNS = [
    ('d0_b1_ref', 'none'),
    ('d1_morph', 'morph_close=5'),
    ('d2_morph_dilate', 'morph+dilate wall:2,door:2,window:1'),
    ('d2_dilate_wall1_door1', 'morph+dilate wall:1,door:1'),
    ('d3_crf_on', 'morph+dilate+crf'),
    ('d4_tta_on', 'tta'),
    ('d5_split_dw', 'morph+dilate+split_door_wall'),
    ('d_best_combo', 'best combo'),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--improve-dir', type=str, default='outputs/e2e_improve')
    return p.parse_args()


def load_run_metrics(improve: Path, name: str) -> dict | None:
    summ = improve / name / 'summary.json'
    if not summ.is_file():
        return None
    data = json.loads(summ.read_text(encoding='utf-8'))
    agg = data.get('aggregate', {})
    if agg.get('num_gt_instances', 0) != 3105:
        return None
    return {
        'name': name,
        'match_rate': agg.get('match_rate'),
        'pred_gt_ratio': agg.get('pred_gt_ratio'),
        'strict_e2e': agg.get('strict_e2e_all_gt'),
        'cls_on_matched': agg.get('e2e_top1_on_matched'),
    }


def check_d_pass(metrics: dict) -> tuple[bool, dict]:
    m = metrics
    gates = {
        'D-PASS-1': m['match_rate'] >= D_GATES['match_rate_ge_61'],
        'D-PASS-2': m['pred_gt_ratio'] <= D_GATES['pred_gt_ratio_le_108'],
        'D-PASS-3': m['strict_e2e'] >= D_GATES['strict_e2e_ge_515'],
        'D-PASS-4': m['cls_on_matched'] >= D_GATES['cls_on_matched_ge_83'],
    }
    d_fail = (
        m['match_rate'] < D_FAIL_MATCH or m['pred_gt_ratio'] > D_FAIL_PRED_GT
    )
    d_pass = all(gates.values()) and not d_fail
    return d_pass, gates


def main():
    args = parse_args()
    improve = Path(args.improve_dir)
    rows = []
    for name, refine in RUNS:
        m = load_run_metrics(improve, name)
        if m is None:
            rows.append({'name': name, 'refine': refine, 'match_rate': None})
            continue
        m['refine'] = refine
        rows.append(m)

    valid = [r for r in rows if r.get('match_rate') is not None]
    best = None
    if valid:
        best = max(
            valid,
            key=lambda r: (
                r['match_rate'],
                -abs(r['pred_gt_ratio'] - 1.05),
                r['strict_e2e'],
            ),
        )

    d_pass = False
    gates = {}
    deploy_refine = None
    if best:
        d_pass, gates = check_d_pass(best)
        if d_pass:
            deploy_refine = best.get('refine')

    out = {
        'baseline_match': BASELINE_MATCH,
        'c_match': C_MATCH,
        'runs': rows,
        'best_run': best,
        'D_PASS': d_pass,
        'D_gates': gates,
        'deploy_refine': deploy_refine,
        'deploy_note': (
            'Replace B1 with D refine stack' if d_pass
            else 'Keep B1 deploy; D did not pass gates'
        ),
    }
    out_path = improve / 'd_plan_summary.json'
    out_path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    print(f'D_PASS={d_pass}  -> {out_path}')


if __name__ == '__main__':
    main()
