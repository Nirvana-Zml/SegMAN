#!/usr/bin/env python3
"""Summarize E2E eval outputs (per_image/*.json) into metrics report."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--eval-dir', type=str, required=True,
                   help='e.g. outputs/e2e_segment_classify/val_full')
    p.add_argument('--out', type=str, default=None,
                   help='Report JSON path (default: eval-dir/e2e_metrics_report.json)')
    return p.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    per_dir = eval_dir / 'per_image'
    if not per_dir.is_dir():
        raise FileNotFoundError(f'Missing {per_dir}')

    files = sorted(per_dir.glob('*.json'))
    if not files:
        raise FileNotFoundError(f'No JSON in {per_dir}')

    total_gt = total_pred = total_matched = 0
    total_correct = total_grasp = total_correct_grasp = 0
    per_class = defaultdict(lambda: {
        'gt_instances': 0, 'matched': 0, 'correct': 0,
        'grasp': 0, 'correct_grasp': 0,
    })
    seg_cls_correct = 0
    seg_cls_on_matched = 0

    for fp in files:
        data = json.loads(fp.read_text(encoding='utf-8'))
        ev = data.get('eval')
        if not ev:
            continue
        total_gt += ev['num_gt_instances']
        total_pred += ev['num_pred_instances']
        total_matched += ev['num_matched']
        for m in ev.get('matches', []):
            gt_c = m['gt_class']
            per_class[gt_c]['gt_instances'] += 1
            if not m.get('matched'):
                continue
            per_class[gt_c]['matched'] += 1
            seg_cls_on_matched += 1
            if m.get('seg_class') == gt_c:
                seg_cls_correct += 1
            if m.get('correct'):
                total_correct += 1
                per_class[gt_c]['correct'] += 1
            if m.get('action') == 'grasp':
                total_grasp += 1
                per_class[gt_c]['grasp'] += 1
                if m.get('correct'):
                    total_correct_grasp += 1
                    per_class[gt_c]['correct_grasp'] += 1

    def rate(num, den):
        return round(num / den, 4) if den > 0 else 0.0

    per_class_report = {}
    for cls, s in sorted(per_class.items()):
        per_class_report[cls] = {
            'gt_instances': s['gt_instances'],
            'matched': s['matched'],
            'match_rate': rate(s['matched'], s['gt_instances']),
            'cls_top1_on_matched': rate(s['correct'], s['matched']),
            'cls_top1_grasp_only': rate(s['correct_grasp'], s['grasp']),
            'grasp_rate': rate(s['grasp'], s['matched']),
        }

    summary_path = eval_dir / 'summary.json'
    base_agg = {}
    if summary_path.is_file():
        base_agg = json.loads(summary_path.read_text(encoding='utf-8')).get('aggregate', {})

    report = {
        'eval_dir': str(eval_dir),
        'num_images': len(files),
        'instance_level': {
            'num_gt_instances': total_gt,
            'num_pred_instances': total_pred,
            'num_matched': total_matched,
            'match_rate': rate(total_matched, total_gt),
            'pred_gt_ratio': rate(total_pred, total_gt),
            'redundancy_excess': total_pred - total_gt,
            'strict_e2e_all_gt': rate(total_correct, total_gt),
            'e2e_top1_on_matched': rate(total_correct, total_matched),
            'e2e_top1_grasp_only': rate(total_correct_grasp, total_grasp),
            'grasp_rate_on_matched': rate(total_grasp, total_matched),
            'seg_class_acc_on_matched': rate(seg_cls_correct, seg_cls_on_matched),
        },
        'per_class_gt_instance': per_class_report,
        'script_aggregate': base_agg,
        'offline_baselines': {
            'gt_roi_top1': 0.7691,
            'segman_roi_top1': 0.6749,
            'note': 'Offline ROI eval on pre-cropped datasets; E2E is instance-matched pipeline.',
        },
    }

    out_path = Path(args.out) if args.out else eval_dir / 'e2e_metrics_report.json'
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    md_path = eval_dir / 'e2e_metrics_report.md'
    lines = [
        '# E2E 评测报告',
        '',
        f'- 评测目录：`{eval_dir}`',
        f'- 图像数：**{len(files)}**',
        '',
        '## 实例级汇总（GT 实例 + mask IoU 匹配）',
        '',
        '| 指标 | 值 |',
        '|------|-----|',
    ]
    il = report['instance_level']
    rows = [
        ('GT 实例总数', il['num_gt_instances']),
        ('预测实例总数', il['num_pred_instances']),
        ('pred/GT 比', f"{il.get('pred_gt_ratio', 0):.4f}"),
        ('冗余 excess (pred−gt)', il.get('redundancy_excess', 0)),
        ('严格端到端 Acc', f"{il.get('strict_e2e_all_gt', 0):.2%}"),
        ('匹配成功数', il['num_matched']),
        ('实例匹配率 match_rate', f"{il['match_rate']:.2%}"),
        ('E2E 分类 Top-1（匹配对上）', f"{il['e2e_top1_on_matched']:.2%}"),
        ('E2E Top-1（仅 grasp）', f"{il['e2e_top1_grasp_only']:.2%}"),
        ('匹配对中 grasp 比例', f"{il['grasp_rate_on_matched']:.2%}"),
        ('分割语义类正确率（匹配对上）', f"{il['seg_class_acc_on_matched']:.2%}"),
    ]
    for k, v in rows:
        lines.append(f'| {k} | {v} |')
    lines.extend([
        '',
        '## 与离线上界对比',
        '',
        '| 评测方式 | Top-1 |',
        '|----------|-------|',
        '| GT-ROI 离线（上界） | 76.91% |',
        '| SegMAN-ROI 离线（部署向） | 67.49% |',
        f"| **E2E 本流水线（匹配对分类）** | **{il['e2e_top1_on_matched']:.2%}** |",
        f"| E2E + 拒识（grasp only） | {il['e2e_top1_grasp_only']:.2%} |",
        '',
        '## 按 GT 类（匹配对上的分类 Acc）',
        '',
        '| 类 | GT实例 | 匹配 | 匹配率 | Cls Acc | Acc(grasp) |',
        '|----|--------|------|--------|---------|------------|',
    ])
    for cls, s in per_class_report.items():
        lines.append(
            f"| {cls} | {s['gt_instances']} | {s['matched']} | "
            f"{s['match_rate']:.2%} | {s['cls_top1_on_matched']:.2%} | "
            f"{s['cls_top1_grasp_only']:.2%} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report['instance_level'], indent=2))
    print(f'Wrote {out_path}')
    print(f'Wrote {md_path}')


if __name__ == '__main__':
    main()
