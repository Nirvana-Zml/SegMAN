#!/usr/bin/env python3
"""Aggregate P0 thresh sweep results and check F1-PASS-4 + constraints."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPROVE = PROJECT_ROOT / 'outputs' / 'e2e_improve'

EXPERIMENTS = [
    ('P0-a', 'f1_m2f_e2e', 0.30, 128, 0.5),  # canonical baseline dir
    ('P0-a', 'f1_m2f_e2e_thresh0.30', 0.30, 128, 0.5),
    ('P0-b', 'f1_m2f_e2e_thresh0.35', 0.35, 128, 0.5),
    ('P0-c', 'f1_m2f_e2e_thresh0.40', 0.40, 128, 0.5),
    ('P0-d', 'f1_m2f_e2e_thresh0.45', 0.45, 128, 0.5),
    ('P0-e', 'f1_m2f_e2e_thresh0.40_ma160', 0.40, 160, 0.5),
    ('P0-f', 'f1_m2f_e2e_thresh0.35_nms06', 0.35, 128, 0.6),
]

GATES = {
    'match_rate_ge_65': 0.65,
    'pred_gt_ratio_min': 0.95,
    'pred_gt_ratio_max': 1.08,
    'strict_e2e_ge_55': 0.55,
    'cls_on_matched_ge_83': 0.83,
    'wall_match_ge_50': 0.50,
}


def load_metrics(out_name: str) -> dict | None:
    summ = IMPROVE / out_name / 'summary.json'
    report = IMPROVE / out_name / 'e2e_metrics_report.json'
    if not summ.is_file():
        return None
    data = json.loads(summ.read_text(encoding='utf-8'))
    agg = data.get('aggregate', {})
    wall_match = None
    if report.is_file():
        pc = json.loads(report.read_text(encoding='utf-8')).get(
            'per_class_gt_instance', {})
        wg = pc.get('wall', {})
        if wg.get('gt_instances'):
            wall_match = wg.get('matched', 0) / wg['gt_instances']
    return {
        'match_rate': agg.get('match_rate'),
        'pred_gt_ratio': agg.get('pred_gt_ratio'),
        'strict_e2e': agg.get('strict_e2e_all_gt'),
        'cls_on_matched': agg.get('e2e_top1_on_matched'),
        'wall_match': round(wall_match, 4) if wall_match is not None else None,
    }


def check_pass(m: dict) -> dict:
    return {
        'F1-PASS-1': m['match_rate'] >= GATES['match_rate_ge_65'],
        'F1-PASS-2': (
            GATES['pred_gt_ratio_min'] <= m['pred_gt_ratio'] <= GATES['pred_gt_ratio_max']
        ),
        'F1-PASS-3': m['strict_e2e'] >= GATES['strict_e2e_ge_55'],
        'F1-PASS-4': m['cls_on_matched'] >= GATES['cls_on_matched_ge_83'],
        'F1-PASS-5': (
            m.get('wall_match') is not None and m['wall_match'] >= GATES['wall_match_ge_50']
        ),
    }


def main():
    seen_dirs = set()
    rows = []
    for eid, out_dir, thresh, min_area, nms in EXPERIMENTS:
        if out_dir in seen_dirs:
            continue
        m = load_metrics(out_dir)
        if m is None:
            continue
        seen_dirs.add(out_dir)
        gates = check_pass(m)
        all_pass = all(gates.values())
        rows.append({
            'id': eid,
            'out_dir': out_dir,
            'm2f_score_thresh': thresh,
            'min_area': min_area,
            'nms_iou': nms,
            **m,
            'gates': gates,
            'all_f1_gates': all_pass,
        })

    best = None
    valid = [r for r in rows if r.get('cls_on_matched') is not None]
    # Prefer full F1 pass, then highest cls with match>=65%
    full_pass = [r for r in valid if r['all_f1_gates']]
    pool = full_pass if full_pass else [
        r for r in valid if r['match_rate'] >= GATES['match_rate_ge_65']
    ]
    if pool:
        best = max(
            pool,
            key=lambda r: (
                r['cls_on_matched'],
                r['match_rate'],
                -abs(r['pred_gt_ratio'] - 1.0),
            ),
        )

    out = {
        'baseline_ref': {
            'out_dir': 'f1_m2f_e2e',
            'cls_on_matched': 0.8178,
            'match_rate': 0.7546,
        },
        'experiments': rows,
        'best': best,
        'recommendation': (
            f"Use {best['out_dir']} with thresh={best['m2f_score_thresh']}"
            if best and best.get('all_f1_gates')
            else (
                f"Best partial: {best['out_dir']} cls={best['cls_on_matched']:.4f}"
                if best
                else 'No completed runs'
            )
        ),
    }
    path = IMPROVE / 'f1_cls_opt_summary.json'
    path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
