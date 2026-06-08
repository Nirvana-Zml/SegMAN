"""Fig 4: 11-class confusion matrix heatmap (GT-ROI)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import paths
from .io_utils import load_json

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def run(out_dir: Path | None = None, confusion_path: Path | None = None) -> Path:
    out_dir = out_dir or paths.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(confusion_path or paths.CONFUSION_GT)
    names = data['class_names']
    matrix = np.array(data['matrix'], dtype=int)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    if HAS_SEABORN:
        sns.heatmap(
            matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=names, yticklabels=names,
            ax=ax, cbar_kws={'label': '样本数'},
            linewidths=0.5, linecolor='white',
        )
    else:
        im = ax.imshow(matrix, cmap='Blues')
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticklabels(names)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0:
                    ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=7)
        fig.colorbar(im, ax=ax, label='样本数')

    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')
    ax.set_title('OpenCLIP 细分类混淆矩阵（GT-ROI val）')

    out = out_dir / '04_confusion_matrix_gt_roi.png'
    fig.savefig(out)
    plt.close(fig)

    row_sum = matrix.sum(axis=1, keepdims=True)
    row_acc = np.divide(np.diag(matrix), row_sum.squeeze(), where=row_sum.squeeze() > 0)
    meta = {
        'class_names': names,
        'per_class_recall_on_diagonal': {
            names[i]: float(row_acc[i]) for i in range(len(names)) if row_sum[i, 0] > 0
        },
        'total_samples': int(matrix.sum()),
    }
    (out_dir / '04_confusion_matrix_meta.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Wrote', out)
    return out


if __name__ == '__main__':
    run()
