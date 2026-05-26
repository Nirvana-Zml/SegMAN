#!/usr/bin/env python3
"""Summarize top off-diagonal confusion pairs (P3-0)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Summarize confusion matrix pairs')
    p.add_argument('--confusion', type=str, required=True)
    p.add_argument('--topk', type=int, default=15)
    p.add_argument('--out', type=str, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(Path(args.confusion).read_text(encoding='utf-8'))
    names = data['class_names']
    matrix = data['matrix']
    pairs = []
    for i, true_name in enumerate(names):
        for j, pred_name in enumerate(names):
            if i == j:
                continue
            count = int(matrix[i][j])
            if count > 0:
                pairs.append({
                    'true': true_name,
                    'pred': pred_name,
                    'count': count,
                    'true_support': int(sum(matrix[i])),
                })
    pairs.sort(key=lambda x: (-x['count'], x['true'], x['pred']))
    top = pairs[: args.topk]
    out = {
        'source': str(Path(args.confusion).resolve()),
        'topk': args.topk,
        'pairs': top,
        'door_wall_total': sum(
            p['count'] for p in pairs
            if {p['true'], p['pred']} == {'door', 'wall'}),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {out_path}  door↔wall total={out["door_wall_total"]}')
    for p in top[:8]:
        print(f"  {p['true']} -> {p['pred']}: {p['count']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
