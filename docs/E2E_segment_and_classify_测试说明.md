# E2E `segment_and_classify` 测试说明

**版本**：v1.0  
**日期**：2026-05-26  
**脚本**：`transgrasp/pipelines/segment_and_classify.py`  
**流程**：原图 → **SegMAN v2@6k** 语义分割 → 连通域裁 ROI → **P3 OpenCLIP** 细分类 → **按类拒识**

---

## 1. 目的与范围


| 项目   | 说明                                                            |
| ---- | ------------------------------------------------------------- |
| 测试对象 | 分割 `iter_6000.pth` + 分类 `deliver_classifier_best.pth`         |
| 输入   | 单张 JPG/PNG，或 Trans10K `img_dir/{train,val}`                   |
| 输出   | 每图 JSON（实例列表 + 置信度 + grasp/reject）；可选 ROI 小图、预测语义 PNG         |
| 评测模式 | `--eval-split val`：按 **mask IoU** 将预测实例与 GT 实例匹配，统计 E2E Top-1 |


**与离线 ROI 评测的区别**：

- 离线 `eval_openclip_classifier.py` 使用 **已裁好的 ROI**（GT 或预导出 SegMAN ROI）。
- 本脚本在 **同一张原图** 上现场跑分割再裁图，用于联调、demo 与 **真实流水线** 精度估计。

---

## 2. 环境与依赖

### 2.1 Docker（推荐）

```bash
docker exec segman_train bash -lc \
  'source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman && \
   cd /workspace/segman && bash scripts/run_e2e_smoke.sh'
```

工作目录：`/workspace/segman`（仓库 `SegMAN/` 挂载）。



### 2.3 必备文件


| 文件          | 默认路径                                                                            |
| ----------- | ------------------------------------------------------------------------------- |
| 分割 config   | `segmentation/local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py` |
| 分割权重        | `segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth`          |
| 分类 deliver  | `outputs/openclip_classifier/deliver_classifier_best.pth`                       |
| 拒识阈值        | `transgrasp/classification/configs/reject_thresholds_p3.json`                   |
| Val 图像      | `segmentation/data/trans10k/img_dir/val/*.jpg`                                  |
| Val GT mask | `segmentation/data/trans10k/ann_dir/val/*.png`                                  |


缺任一文件时脚本会报 `FileNotFoundError`，请先完成分割训练与 P3 deliver 升级。

---

## 3. 快速测试（约 1 分钟）

### 3.1 单图冒烟

```bash
bash scripts/run_e2e_smoke.sh
```

或手动：

```bash
python transgrasp/pipelines/segment_and_classify.py \
  --image segmentation/data/trans10k/img_dir/val/val_000000.jpg \
  --out-dir outputs/e2e_segment_classify/smoke \
  --save-rois \
  --save-sem-seg \
  --device cuda:0
```

**验收**：

1. 终端打印 JSON，含 `instances` 列表（每项含 `pred_class`、`confidence`、`action`）。
2. 生成 `outputs/e2e_segment_classify/smoke/val_000000.json`。
3. 若加 `--save-sem-seg`：`sem_seg_pred/val_000000.png`（class-id 灰度图，目视偏暗属正常）。
4. 若加 `--save-rois`：`roi_crops/val_000000/*.jpg` 与预测类名、grasp/reject 一致。

### 3.2 单图输出字段说明

```json
{
  "image_stem": "val_000000",
  "num_pred_instances": 3,
  "instances": [
    {
      "class_id": 8,
      "class_name": "cup",
      "bbox": [120, 40, 280, 200],
      "pred_class": "cup",
      "confidence": 0.91,
      "threshold": 0.45,
      "action": "grasp",
      "topk": [{"class": "cup", "confidence": 0.91}, ...]
    }
  ]
}
```

- `class_name`（分割）：SegMAN 对该连通域的 **语义类**（裁 ROI 时用该类 bbox）。
- `pred_class`（分类）：OpenCLIP 细分类结果（可与分割类不同，匹配评测以 GT 为准）。
- `action`：`grasp` = 置信度 ≥ 该类阈值；`reject` = 建议人工确认。

---

## 4. Val 集 E2E 评测（约 10～30 分钟 / 100 张）

### 4.1 默认 100 张

```bash
bash scripts/run_e2e_eval_val.sh
```

环境变量：

```bash
MAX_IMAGES=50 bash scripts/run_e2e_eval_val.sh    # 50 张
MAX_IMAGES=-1 bash scripts/run_e2e_eval_val.sh   # 全量 val（约 1000 张，较慢）
```

### 4.2 手动命令

```bash
python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val \
  --data-root segmentation/data/trans10k \
  --max-images 100 \
  --out-dir outputs/e2e_segment_classify/val_100 \
  --iou-match 0.3 \
  --device cuda:0
```

### 4.3 评测逻辑

对每张图：

1. SegMAN 预测语义图 → 提取预测实例（与 `build_roi_dataset` 相同：`bbox-pad=0.15`，`min-area=64`）。
2. 每个预测 ROI 送 OpenCLIP 分类 + 拒识。
3. 从 **GT ann** 提取 GT 实例；预测实例与 GT 实例做 **贪心 mask IoU 匹配**（默认 IoU ≥ 0.3）。
4. 在匹配对上统计：`top1_on_matched`（分类是否正确）、`accept_rate_on_matched`（grasp 比例）。

**汇总文件**：


| 文件                      | 内容                     |
| ----------------------- | ---------------------- |
| `summary.json`          | `aggregate` 全 val 汇总指标 |
| `per_image_metrics.csv` | 每张图一行                  |
| `per_image/{stem}.json` | 单图明细 + `eval.matches`  |


### 4.4 指标解读


| 指标                    | 含义                                              | 参考                              |
| --------------------- | ----------------------------------------------- | ------------------------------- |
| `match_rate`          | GT 实例中被预测 mask 匹配上的比例                           | 分割漏检/碎裂会降低                      |
| `e2e_top1_on_matched` | 匹配对上的分类 Top-1                                   | 应接近 SegMAN-ROI 离线 ~67% 量级（全流水线） |
| `e2e_top1_grasp_only` | 仅统计 `action=grasp` 的匹配对                         | 对应方案 B 有效决策精度                   |
| 离线 GT-ROI Top-1       | `eval_openclip_classifier` on `trans10k_roi_gt` | **76.91%**（上界，非 E2E）            |


E2E 低于离线 ROI 是正常现象：**误差传递**（错 mask、错裁、实例未匹配）。

---

## 5. 自定义图片目录

```bash
python transgrasp/pipelines/segment_and_classify.py \
  --image-dir /path/to/my_images \
  --out-dir outputs/e2e_segment_classify/custom \
  --device cuda:0
```

无 GT 时不输出 `eval` 字段，仅输出预测 JSON。

---

## 6. 参数一览


| 参数                   | 默认                            | 说明                |
| -------------------- | ----------------------------- | ----------------- |
| `--seg-config`       | v2 balanced config            | 分割结构              |
| `--seg-checkpoint`   | `iter_6000.pth`               | 分割权重              |
| `--cls-checkpoint`   | `deliver_classifier_best.pth` | P3 分类             |
| `--class-thresholds` | `reject_thresholds_p3.json`   | 按类拒识 τ            |
| `--bbox-pad`         | 0.15                          | 与 ROI 数据集一致       |
| `--min-area`         | 64                            | 最小连通域像素           |
| `--iou-match`        | 0.3                           | GT–预测实例匹配阈值       |
| `--max-images`       | -1                            | 限制张数（调试）          |
| `--save-rois`        | off                           | 保存 ROI 小图         |
| `--save-sem-seg`     | off                           | 保存预测 class-id PNG |


---

## 7. 常见问题

### Q1：`ModuleNotFoundError: mmseg`

确保在 `segman` 环境中运行，且 `segmentation/` 在仓库内完整。脚本会自动把 `segmentation/` 加入 `sys.path`。

### Q2：语义 PNG 全黑

`sem_seg_pred/*.png` 存的是 **类别 id（0–11）**，不是彩色可视化。请查看 `roi_crops/` 或原图叠加工具。

### Q3：E2E 准确率远低于 76%

预期行为。76.91% 是 **GT mask 裁 ROI** 的上界；E2E 含分割与实例匹配误差。对比：

- `data/trans10k_roi_segman` + `eval_openclip_classifier` → 部署向离线 **67.49%**
- 本脚本 E2E `e2e_top1_on_matched` → 应与离线 SegMAN-ROI 同量级或略低

### Q4：CUDA OOM

单图推理显存通常 < 4GB。若 OOM，使用 `--device cpu`（较慢）。

### Q5：Windows 路径

在 PowerShell 中 `cd` 到 `SegMAN` 后执行同样命令；Docker 方式与 Linux 一致。

---

## 8. 相关脚本与模块


| 路径                                                | 作用         |
| ------------------------------------------------- | ---------- |
| `transgrasp/pipelines/segment_and_classify.py`    | E2E 主入口    |
| `transgrasp/pipelines/seg_model.py`               | MMSeg 封装   |
| `transgrasp/pipelines/roi_extract.py`             | 连通域裁 ROI   |
| `transgrasp/pipelines/classify_instances.py`      | 批量 ROI 分类  |
| `scripts/run_e2e_smoke.sh`                        | 单图冒烟       |
| `scripts/run_e2e_eval_val.sh`                     | Val 子集评测   |
| `transgrasp/pipelines/summarize_e2e_eval.py`      | 汇总 per-class 报告 |
| `transgrasp/classification/infer_openclip_roi.py` | 仅分类（单 ROI） |
| `transgrasp/data/build_roi_dataset.py`            | 离线批量导 ROI  |


---

## 9. 建议测试记录表（答辩 / 验收用）


| 步骤  | 命令                                   | 预期                                            | 实际  | 通过  |
| --- | ------------------------------------ | --------------------------------------------- | --- | --- |
| 1   | `run_e2e_smoke.sh`                   | 生成 `smoke/*.json`，instances≥1                 | val_000000：7 实例 | ✅   |
| 2   | 目视 `roi_crops/`                      | crop 含透明物体主体                                  | 已生成 ROI 小图 | ✅   |
| 3   | 全量 val（1000 张）                       | `summary.json` 有 `aggregate`                  | `val_full/summary.json` | ✅   |
| 4   | 对比离线 SegMAN-ROI                      | 匹配对上分类 Acc 与离线同量级                          | 84.09%（匹配对） | ✅   |
| 5   | 拒识                                   | `e2e_top1_grasp_only` > `e2e_top1_on_matched` | 90.83% > 84.09% | ✅   |

---

## 10. 实测结果（2026-05-26）

**环境**：Docker `segman_train`，conda `segman`  
**命令**：

```bash
bash scripts/run_e2e_smoke.sh
python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --out-dir outputs/e2e_segment_classify/val_full
python transgrasp/pipelines/summarize_e2e_eval.py \
  --eval-dir outputs/e2e_segment_classify/val_full
```

**产物目录**：`outputs/e2e_segment_classify/val_full/`  
（`summary.json`、`per_image_metrics.csv`、`e2e_metrics_report.md`）

### 10.1 实例级汇总（1000 张 val，3105 GT 实例）

| 指标 | 数值 | 说明 |
|------|------|------|
| 预测实例总数 | 3563 | SegMAN 连通域数（含过分割） |
| GT 实例匹配率 | **59.32%** | 1842 / 3105 个 GT 实例被预测 mask 匹配（IoU≥0.3） |
| **E2E 分类 Top-1（匹配对上）** | **84.09%** | 在成功匹配的实例上，细分类正确率 |
| **E2E + 拒识（grasp only）** | **90.83%** | 仅高置信 grasp 决策的正确率 |
| 匹配对中 grasp 比例 | 79.37% | 约 20% 匹配实例建议拒识 |
| 分割语义类正确率（匹配对上） | 92.94% | 预测 mask 的语义类 vs GT 类 |
| **端到端全 GT 召回 Acc** | **≈49.9%** | 1549 / 3105（匹配且分类正确 / 全部 GT 实例） |

> **指标解读**：84.09% 高于离线 SegMAN-ROI **67.49%**，是因为 E2E 评测只在「匹配成功」的子集上算分类 Acc（偏易样本）；真实部署应同时看 **匹配率 59%** 与 **全 GT 端到端 ≈50%**。离线 SegMAN-ROI 67.49% 对应「已裁好的 ROI 集合」，与 E2E 统计口径不同。

### 10.2 与离线上界对比

| 评测方式 | Top-1 | 样本/口径 |
|----------|-------|-----------|
| GT-ROI 离线（分类上界） | **76.91%** | 3105 ROI，GT mask 裁切 |
| SegMAN-ROI 离线（部署向） | **67.49%** | 3233 ROI，预导出 v2@6k mask |
| E2E 匹配对分类 | **84.09%** | 1842 匹配 GT 实例 |
| E2E + 拒识 grasp | **90.83%** | 1462 grasp 实例 |
| E2E 全 GT 实例（严格） | **≈49.9%** | 3105 GT 实例 |

### 10.3 按类（匹配对上的分类 Acc）

| 类 | GT 实例 | 匹配率 | Cls Acc | Acc (grasp) | 备注 |
|----|---------|--------|---------|-------------|------|
| eyeglass | 92 | 88.04% | **100%** | 100% | 强类 |
| cup | 366 | 94.26% | **96.23%** | 97.34% | 强类 |
| wall | 1290 | 48.14% | 88.08% | 94.07% | 匹配率低、类内 Acc 高 |
| bottle | 223 | 78.48% | 88.00% | 95.48% | |
| bowl | 38 | 76.32% | 86.21% | 95.65% | |
| box | 88 | 64.77% | 78.95% | 86.49% | |
| freezer | 20 | 70.00% | 85.71% | 100% | |
| jar_kettle | 133 | 79.70% | 75.47% | 80.00% | |
| door | 663 | **45.85%** | **69.41%** | 76.84% | 弱类：匹配+分类双低 |
| window | 130 | 58.46% | **59.21%** | 71.15% | 弱类 |
| shelf | 62 | 54.84% | **50.00%** | 63.64% | 弱类 |

**瓶颈**：door / wall / window / shelf 的 **实例匹配率** 与 **分类 Acc** 均偏低，与 P0～P3 审计结论一致。

> **改进方案**：详见 [E2E_性能分析与改进方案.md](./E2E_性能分析与改进方案.md)（E1 后处理 → E2 分割弱类 → E3 结构类分类 → E4 回归/demo）。

---

## 11. 修订记录


| 日期         | 版本   | 说明                          |
| ---------- | ---- | --------------------------- |
| 2026-05-26 | v1.0 | 首版：E2E 脚本 + 冒烟/val 评测 + 本文档 |
| 2026-05-26 | v1.1 | §10 填入全量 val 1000 张实测结果 |


