#!/usr/bin/env python3
"""E2-0: export unmatched GT instances + heuristic root-cause + audit artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw

DOOR_WALL = {'door', 'wall'}


def parse_args():
    p = argparse.ArgumentParser(description='Export E2-0 unmatched audit')
    p.add_argument('--eval-dir', type=str, required=True)
    p.add_argument('--out-dir', type=str, default='outputs/e2e_improve/e2_audit')
    p.add_argument('--data-root', type=str, default='segmentation/data/trans10k')
    p.add_argument('--sample-wall', type=int, default=100)
    p.add_argument('--sample-door', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--render-vis', action='store_true')
    return p.parse_args()


def infer_root_cause(m: dict) -> str:
    """Heuristic per §7.1 (miss > adhesion > fragment > iou_gap > class_swap)."""
    iou = float(m.get('match_iou', 0))
    gt = m['gt_class']
    seg = m.get('best_seg_class', '')
    overlap = int(m.get('pred_overlap_count', 0))

    if iou < 0.10:
        return 'miss'
    if gt in DOOR_WALL and seg in DOOR_WALL and gt != seg:
        return 'adhesion'
    if overlap >= 2 and iou < 0.30:
        return 'fragment'
    if seg == gt and 0.10 <= iou < 0.30:
        return 'iou_gap'
    if seg and seg != gt and iou >= 0.10:
        return 'class_swap'
    if iou < 0.30:
        return 'iou_gap'
    return 'miss'


def resolve_mmseg_root(data_root: str) -> Path:
    p = Path(data_root)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def render_triple(img_path: Path, ann_path: Path, stem: str, gt_class: str,
                  bbox: list, out_path: Path):
    rgb = np.array(Image.open(img_path).convert('RGB'))
    gt = np.array(Image.open(ann_path))
    if gt.ndim == 3:
        gt = gt[..., 0]
    h, w = rgb.shape[:2]

    gt_vis = rgb.copy()
    mask = gt > 0
    gt_vis[mask] = (gt_vis[mask] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)

    pred_vis = rgb.copy()
    x0, y0, x1, y1 = bbox
    draw = ImageDraw.Draw(Image.fromarray(pred_vis))
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
    draw.text((x0, max(0, y0 - 12)), gt_class, fill=(255, 0, 0))

    panel = np.concatenate([rgb, gt_vis, pred_vis], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(out_path, quality=90)


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = PROJECT_ROOT / eval_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mmseg = resolve_mmseg_root(args.data_root)
    img_dir = mmseg / 'img_dir' / 'val'
    ann_dir = mmseg / 'ann_dir' / 'val'

    per_dir = eval_dir / 'per_image'
    rows = []
    by_class = Counter()
    by_cause = Counter()
    by_class_cause: dict[str, Counter] = defaultdict(Counter)

    for fp in sorted(per_dir.glob('*.json')):
        data = json.loads(fp.read_text(encoding='utf-8'))
        ev = data.get('eval')
        if not ev:
            continue
        stem = data['image_stem']
        for idx, m in enumerate(ev.get('matches', [])):
            if m.get('matched'):
                continue
            cause = infer_root_cause(m)
            row = {
                'stem': stem,
                'gt_class': m['gt_class'],
                'match_iou': m.get('match_iou', 0),
                'best_seg_class': m.get('best_seg_class', ''),
                'best_pred_class': m.get('best_pred_class', ''),
                'pred_overlap_count': m.get('pred_overlap_count', 0),
                'heuristic_root_cause': cause,
                'match_idx': idx,
            }
            rows.append(row)
            by_class[m['gt_class']] += 1
            by_cause[cause] += 1
            by_class_cause[m['gt_class']][cause] += 1

    csv_path = out_dir / 'unmatched_gt_instances.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            'stem', 'gt_class', 'match_iou', 'heuristic_root_cause'])
        w.writeheader()
        w.writerows(rows)

    unmatched_by_class = dict(by_class)
    (out_dir / 'unmatched_by_class.json').write_text(
        json.dumps(unmatched_by_class, indent=2) + '\n', encoding='utf-8')

    matrix_rows = []
    for cls in sorted(by_class_cause):
        for cause, cnt in sorted(by_class_cause[cls].items()):
            matrix_rows.append({'gt_class': cls, 'root_cause': cause, 'count': cnt})
    with (out_dir / 'e2_root_cause_matrix.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['gt_class', 'root_cause', 'count'])
        w.writeheader()
        w.writerows(matrix_rows)

    rng = random.Random(args.seed)
    samples = []
    for cls, n in [('wall', args.sample_wall), ('door', args.sample_door)]:
        pool = [r for r in rows if r['gt_class'] == cls]
        picks = rng.sample(pool, min(n, len(pool)))
        samples.extend(picks)

    sample_csv = out_dir / 'sample_list.csv'
    with sample_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(samples)

    stems_wall_door = sorted({r['stem'] for r in rows if r['gt_class'] in DOOR_WALL})
    (out_dir / 'candidate_stems_wall_door.txt').write_text(
        '\n'.join(stems_wall_door) + '\n', encoding='utf-8')

    if args.render_vis and samples:
        vis_dir = out_dir / 'vis'
        for s in samples[: min(30, len(samples))]:
            stem = s['stem']
            img_path = img_dir / f'{stem}.jpg'
            if not img_path.is_file():
                img_path = img_dir / f'{stem}.png'
            ann_path = ann_dir / f'{stem}.png'
            if img_path.is_file() and ann_path.is_file():
                out_v = vis_dir / f"{stem}_{s['gt_class']}_{s['heuristic_root_cause']}.jpg"
                render_triple(img_path, ann_path, stem, s['gt_class'], [0, 0, 0, 0], out_v)

    total_unmatched = len(rows)
    wall_causes = by_class_cause.get('wall', Counter())
    door_causes = by_class_cause.get('door', Counter())
    dominant = by_cause.most_common(1)[0][0] if by_cause else 'miss'

    action_lines = [
        '# E2 Action Decision (auto from heuristic E2-0)',
        '',
        f'- Total unmatched GT: **{total_unmatched}**',
        f'- Dominant root cause (all classes): **{dominant}** ({by_cause[dominant]/max(total_unmatched,1):.1%})',
        '',
        '## wall root cause',
        '',
    ]
    for c, n in wall_causes.most_common():
        action_lines.append(f'- {c}: {n} ({n/max(unmatched_by_class.get("wall",1),1):.1%})')
    action_lines.extend(['', '## door root cause', ''])
    for c, n in door_causes.most_common():
        action_lines.append(f'- {c}: {n} ({n/max(unmatched_by_class.get("door",1),1):.1%})')

    rec = []
    if wall_causes.get('adhesion', 0) + door_causes.get('adhesion', 0) >= 0.25 * (
            unmatched_by_class.get('wall', 0) + unmatched_by_class.get('door', 0)):
        rec.append('1. **P0-4c / boundary_loss_weight↑**（door–wall 粘连主导）')
    if wall_causes.get('fragment', 0) + door_causes.get('fragment', 0) >= 0.20 * total_unmatched:
        rec.append('2. **CC merge 后处理 + boundary loss**（碎裂主导）')
    if by_cause.get('miss', 0) >= 0.35 * total_unmatched:
        rec.append('3. **弱类 class_weight↑ + Copy-Paste**（漏检主导）')
    if by_cause.get('iou_gap', 0) >= 0.25 * total_unmatched:
        rec.append('4. **E1 iou-match 微调**（IoU 不足主导，优先非重训）')
    if not rec:
        rec.append('1. 保守 **e2weak finetune**（lr 5e-6, 2000 iter）+ boundary_weight 0.22')

    action_lines.extend(['', '## Recommended E2-1 changes (max 2)', ''] + rec)
    (out_dir / 'e2_action_decision.md').write_text('\n'.join(action_lines) + '\n', encoding='utf-8')

    audit_md = [
        '# E2-0 Unmatched Audit',
        '',
        f'- eval_dir: `{eval_dir}`',
        f'- unmatched GT instances: **{total_unmatched}**',
        '',
        '## By class',
        '',
        '| class | count |',
        '|-------|-------|',
    ]
    for cls, n in sorted(unmatched_by_class.items(), key=lambda x: -x[1]):
        audit_md.append(f'| {cls} | {n} |')
    audit_md.extend(['', '## By root cause (heuristic)', '', '| cause | count | pct |', '|-------|-------|-----|'])
    for cause, n in by_cause.most_common():
        audit_md.append(f'| {cause} | {n} | {n/max(total_unmatched,1):.1%} |')
    (out_dir / 'e2_audit_unmatched.md').write_text('\n'.join(audit_md) + '\n', encoding='utf-8')

    print(f'Unmatched: {total_unmatched}  dominant_cause: {dominant}')
    print(f'Wrote {out_dir}')


if __name__ == '__main__':
    main()
