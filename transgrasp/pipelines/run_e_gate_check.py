#!/usr/bin/env python3
"""Check scheme E gates and write e_plan_summary.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASELINE_MATCH = 0.5916
D_BEST_TTA = 0.6048

E_GATES = {
    'match_rate_ge_65': 0.65,
    'pred_gt_ratio_min': 0.95,
    'pred_gt_ratio_max': 1.08,
    'strict_e2e_ge_55': 0.55,
    'cls_on_matched_ge_83': 0.83,
    'wall_match_ge_55': 0.55,
}

E_FAIL_MATCH = 0.62

RUNS = [
    ('e0_b1_ref', 'semantic'),
    ('e2_gt_oracle', 'gt_oracle'),
    ('e4_maskrcnn_e2e', 'maskrcnn'),
    ('e4_maskrcnn_b1', 'maskrcnn+b1'),
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
    if agg.get('num_gt_instances', 0) != 3105:
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
    }


def check_e_pass(metrics: dict) -> tuple[bool, dict]:
    m = metrics
    gates = {
        'E-PASS-1': m['match_rate'] >= E_GATES['match_rate_ge_65'],
        'E-PASS-2': (
            E_GATES['pred_gt_ratio_min'] <= m['pred_gt_ratio'] <= E_GATES['pred_gt_ratio_max']
        ),
        'E-PASS-3': m['strict_e2e'] >= E_GATES['strict_e2e_ge_55'],
        'E-PASS-4': m['cls_on_matched'] >= E_GATES['cls_on_matched_ge_83'],
        'E-PASS-5': (
            m.get('wall_match') is not None and m['wall_match'] >= E_GATES['wall_match_ge_55']
        ),
    }
    e_fail = m['match_rate'] < E_FAIL_MATCH
    e_pass = all(gates.values()) and not e_fail
    return e_pass, gates


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
    # Exclude gt_oracle from deploy best (upper bound only)
    deploy_candidates = [r for r in valid if r['name'] != 'e2_gt_oracle']
    best = None
    if deploy_candidates:
        best = max(
            deploy_candidates,
            key=lambda r: (
                r['match_rate'],
                -abs(r['pred_gt_ratio'] - 1.0),
                r['strict_e2e'],
            ),
        )

    e_pass = False
    gates = {}
    deploy_source = None
    if best:
        e_pass, gates = check_e_pass(best)
        if e_pass:
            deploy_source = best.get('instance_source')

    out = {
        'baseline_match': BASELINE_MATCH,
        'd_best_tta_match': D_BEST_TTA,
        'runs': rows,
        'best_run': best,
        'E_PASS': e_pass,
        'E_gates': gates,
        'deploy_instance_source': deploy_source,
        'deploy_note': (
            'Replace semantic CC with instance model' if e_pass
            else 'Keep B1 semantic deploy; E did not pass gates'
        ),
    }
    out_path = improve / 'e_plan_summary.json'
    out_path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    print(f'E_PASS={e_pass}  -> {out_path}')


if __name__ == '__main__':
    main()
