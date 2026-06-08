# viz_deliver — 交付数据可视化

从项目已有 JSON 评测结果生成 PNG 图表，用于答辩、软著说明书与报告。

## 依赖

```bash
pip install matplotlib numpy
# 可选（混淆矩阵更好看）
pip install seaborn
```

## 一键生成全部图表

在 `SegMAN/` 根目录执行：

```bash
python viz_deliver/run_all.py
```

输出目录：`viz_deliver/output/`

## 单独运行

```bash
python -m viz_deliver.plot_01_dual_track_e2e
python -m viz_deliver.plot_02_per_class_match
python -m viz_deliver.plot_03_coverage_accuracy
python -m viz_deliver.plot_04_confusion_matrix
python -m viz_deliver.plot_05_per_class_prf
```

或只生成部分图：

```bash
python viz_deliver/run_all.py --only 1 2 3
```

## 图表与数据源

| 输出文件 | 脚本 | 数据源 |
| --- | --- | --- |
| `01_dual_track_e2e_bar.png` | plot_01 | `outputs/e2e_improve/f1_plan_summary.json` |
| `01_dual_track_e2e_radar.png` | plot_01 | 同上 |
| `02_per_class_match_rate.png` | plot_02 | `f1_b1_ref` + `f1_m2f_e2e` 的 `e2e_metrics_report.json` |
| `03_coverage_accuracy.png` | plot_03 | `plan_b/coverage_gt` + `coverage_segman` |
| `04_confusion_matrix_gt_roi.png` | plot_04 | `deliver_p3/eval_gt_roi/confusion_matrix.json` |
| `05_per_class_prf.png` | plot_05 | `eval_gt_roi` + `eval_segman_roi` 的 `per_class_report.json` |

每张图会附带 `*_meta.json` 摘要（部分图）。
