#!/usr/bin/env python3
"""Check scheme F1 gates and write f1_plan_summary.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASELINE_MATCH = 0.5916
ORACLE_MATCH = 0.9298

F1_GATES = {
    'match_rate_ge_65': 0.65,
    'pred_gt_ratio_min': 0.95,
    'pred_gt_ratio_max': 1.08,
    'strict_e2e_ge_55': 0.55,
    'cls_on_matched_ge_83': 0.83,
    'wall_match_ge_50': 0.50,
}

F1_FAIL_MATCH = 0.62

RUNS = [
    ('f1_b1_ref', 'semantic'),
    ('f1_m2f_e2e', 'm2f'),
    ('f1_m2f_smoke50', 'm2f'),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--improve-dir', type=str, default='outputs/e2e_improve')
    return p.parse_args()


def load_run_metrics(improve: Path, name: str) -> dict | None:
    summ = improve / name / 'summary.json'
    report = improve / name / 'e2e_metrics_report.json'
    if not summ.is_file():
        return None
    data = json.loads(summ.read_text(encoding='utf-8'))
    agg = data.get('aggregate', {})
    n_gt = agg.get('num_gt_instances', 0)
    if n_gt not in (3105, 0) and n_gt < 3000:
        return None
    wall_match = None
    if report.is_file():
        pc = json.loads(report.read_text(encoding='utf-8')).get('per_class_gt_instance', {})
        wg = pc.get('wall', {})
        if wg.get('gt_instances'):
            wall_match = wg.get('matched', 0) / wg['gt_instances']
    return {
        'name': name,
        'match_rate': agg.get('match_rate'),
        'pred_gt_ratio': agg.get('pred_gt_ratio'),
        'strict_e2e': agg.get('strict_e2e_all_gt'),
        'cls_on_matched': agg.get('e2e_top1_on_matched'),
        'wall_match': round(wall_match, 4) if wall_match is not None else None,
        'num_gt_instances': n_gt,
    }


def check_f1_pass(metrics: dict) -> tuple[bool, dict]:
    m = metrics
    gates = {
        'F1-PASS-1': m['match_rate'] >= F1_GATES['match_rate_ge_65'],
        'F1-PASS-2': (
            F1_GATES['pred_gt_ratio_min'] <= m['pred_gt_ratio'] <= F1_GATES['pred_gt_ratio_max']
        ),
        'F1-PASS-3': m['strict_e2e'] >= F1_GATES['strict_e2e_ge_55'],
        'F1-PASS-4': m['cls_on_matched'] >= F1_GATES['cls_on_matched_ge_83'],
        'F1-PASS-5': (
            m.get('wall_match') is not None and m['wall_match'] >= F1_GATES['wall_match_ge_50']
        ),
    }
    f1_fail = m['match_rate'] < F1_FAIL_MATCH
    f1_pass = all(gates.values()) and not f1_fail
    return f1_pass, gates


def main():
    args = parse_args()
    improve = Path(args.improve_dir)
    rows = []
    for name, source in RUNS:
        m = load_run_metrics(improve, name)
        if m is None:
            rows.append({'name': name, 'instance_source': source, 'match_rate': None})
            continue
        m['instance_source'] = source
        rows.append(m)

    valid = [r for r in rows if r.get('match_rate') is not None]
    deploy_candidates = [r for r in valid if r['name'] != 'f1_b1_ref' or len(valid) == 1]
    best = None
    if deploy_candidates:
        best = max(
            deploy_candidates,
            key=lambda r: (
                r['match_rate'],
                -abs(r['pred_gt_ratio'] - 1.0),
                r.get('strict_e2e') or 0,
            ),
        )

    f1_pass = False
    gates = {}
    deploy_source = None
    if best and best['name'] != 'f1_b1_ref':
        f1_pass, gates = check_f1_pass(best)
        if f1_pass:
            deploy_source = best.get('instance_source')

    out = {
        'baseline_match': BASELINE_MATCH,
        'oracle_match': ORACLE_MATCH,
        'runs': rows,
        'best_run': best,
        'F1_PASS': f1_pass,
        'F1_gates': gates,
        'deploy_instance_source': deploy_source,
        'deploy_note': (
            'Replace semantic CC with Mask2Former' if f1_pass
            else 'Keep B1 semantic deploy; F1 did not pass gates'
        ),
    }
    out_path = improve / 'f1_plan_summary.json'
    out_path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    print(f'F1_PASS={f1_pass}  -> {out_path}')


if __name__ == '__main__':
    main()
