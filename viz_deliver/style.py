"""Shared matplotlib style for deliver figures."""
from __future__ import annotations

import matplotlib.pyplot as plt

# Chinese labels on Windows; fall back to DejaVu if missing.
plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

COLOR_A = '#4C78A8'   # mode A semantic
COLOR_B = '#F58518'   # mode B grasp
COLOR_GT = '#54A24B'
COLOR_SEG = '#E45756'
