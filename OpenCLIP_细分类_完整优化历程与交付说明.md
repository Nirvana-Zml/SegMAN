# OpenCLIP 透明物体 ROI 细分类 — 完整优化历程与交付说明


| 项目 | 内容 |
|------|------|
| 文档版本 | v1.1 |
| 编写日期 | 2026-05-26 |
| 关联文档 | 《OpenCLIP_细分类_未达80%原因与优化方案.md》《OpenCLIP_细分类训练与优化指南.md》《路线C_细分类与抓取实施步骤.md》 |
| 分割交付 | SegMAN v2@6k，`iter_6000.pth`，mIoU **81.80%** |
| **分类正式交付** | `outputs/openclip_classifier/deliver_classifier_best.pth`（**P3**，2026-05-26 升级） |
| 课题验收目标 | GT-ROI val Top-1 Acc **≥ 80%** |
| **当前交付指标** | GT-ROI **76.91%** / SegMAN-ROI **67.49%** |
| **方案 B 高置信子集** | GT @60% coverage **89.16%** / SegMAN @60% **80.67%** ✅ |
| 距 80% 差距 | **−3.09 pt**（GT-ROI 全局 Top-1） |

---

## 1. 项目背景与验收口径

### 1.1 任务定义

在 Trans10K 透明物体数据集上，基于 SegMAN 分割 mask 裁切 ROI，使用 OpenCLIP（ViT-B-16）对 **11 类** 实例进行 fine-grained 分类。

**双评测集**：

| 评测集 | 含义 | val 样本数 |
|--------|------|------------|
| **GT-ROI** | GT mask 裁切 ROI | 3,105 |
| **SegMAN-ROI** | v2@6k 预测 mask 裁切 ROI | 3,233 |

GT-ROI 反映分类**理论上界**；SegMAN-ROI 反映**部署真实精度**。

### 1.2 固定组件（全实验不变）

| 组件 | 路径 |
|------|------|
| 分割 checkpoint | `segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth` |
| GT ROI | `data/trans10k_roi_gt/`（bbox-pad 0.15） |
| SegMAN ROI | `data/trans10k_roi_segman/val` |
| OpenCLIP 骨干 | ViT-B-16，`laion2b_s34b_b88k` |
| 类数 | 11（box, bottle, window, eyeglass, freezer, jar_kettle, door, cup, wall, bowl, shelf） |

### 1.3 优化路线总览

```text
基线 T1/T2（74.88% GT）
  → P0 分割弱类 finetune          ❌ SegMAN-ROI 下降
  → P1 T2 加深 + encoder 持久化   ✅ GT 74.91% / SegMAN 65.73%
  → P2 层次级联分类               ⚠️ GT 75.23% / SegMAN 67.09%
  → P3 hard mining + 增广         ✅ GT 76.91% / SegMAN 67.49% → 现为 deliver
  → P3+P2 级联 ablation            ❌ GT 75.27%
  → P4 WiSE-FT + contrastive      ❌ 最高 77.04%（small），未达 78%
  → P4-full 全量 contrastive      ❌ GT 76.91%（未超 P3）
  → 正式 deliver 自 T2 升级为 P3   ✅ 2026-05-26
  → 方案 B（拒识 + 双指标结题）     ✅ GT @60% cov **89.16%**；按类拒识 **86.08%**
```

---

## 2. 基线阶段（T1 → T2）

### 2.1 实验结论摘要

| 阶段 | 方法 | GT-ROI | 结论 |
|------|------|--------|------|
| T1 | 冻结 ViT-B + linear + class_weights | 70.18% | baseline |
| §8 sweep | 去 class_weights 等 | 72.62% | class_weights 有害 |
| **T2** | 解冻末 **2** block + linear | **74.88%** | **首版正式 deliver** |
| ViT-L-14 冻结 | 更大 backbone | 72.37% | 无效 |
| bbox-pad 0.10 | 更紧 ROI | 70.82% | 有害 |

**T2 归档指标**（`deliver_t2_best/`）：

- GT-ROI：**74.88%**，macro-F1 72.19%
- SegMAN-ROI：**64.61%**，macro-F1 58.37%
- ΔAcc（GT − SegMAN）：**−10.27 pt**

**主要弱类**（T2 GT-ROI）：door F1 **63.5%**，window **51.2%**，shelf **51.0%**；door↔wall 结构混淆占错分主体。

---

## 3. P0 — 分割弱类专项（❌ 未通过）

### 3.1 目的

在不改分类 checkpoint 的前提下，通过改善 SegMAN 弱类 mask，提升 SegMAN-ROI Acc。

### 3.2 实施过程

1. **P0-0 基线审计**：per-class IoU + ROI 级 GT/SegMAN F1 落差表（见 `outputs/p0_weak_audit.md`）。
2. **P0-1 弱类 CE finetune**：从 v2@6k 热启动，4000 iter，提高 shelf/door/box/freezer/bowl 权重。
3. **P0-2 重导 ROI**：`data/trans10k_roi_segman_p0weak/val`（3443 ROIs，+210 vs 基线）。
4. **P0-3 闸门**：固定 T2 分类器，仅换 SegMAN ROI 重评。

### 3.3 结果

| 指标 | v2@6k 基线 | P0 (iter_2000 mask) | 判定 |
|------|------------|---------------------|------|
| SegMAN-ROI Acc | **64.61%** | **61.81%** | ❌ 回退 |
| macro F1 | 58.37% | 56.50% | ❌ |
| mIoU | 81.80% | 81.00% | 通过 |

**决策**：**保留 v2@6k 分割**；P0 不更新交付。审计报告：`outputs/p0_weak_audit.md`。

---

## 4. P1 — T2 加深 + Encoder 持久化（✅ 通过）

### 4.1 目的

解冻更多 ViT block，保存 encoder 权重，提升 GT-ROI 与 SegMAN-ROI。

### 4.2 关键发现

**直接解冻 4 block 失败**；必须 **两阶段**：

| 阶段 | 配置 | 结果 |
|------|------|------|
| Stage-1 warmup | 解冻 **2** block，从 T2 resume | val 最高 73.69%，**写出 encoder** |
| Stage-2 deepen | 解冻 **4** block，从 stage-1 resume | **GT 74.91%**，SegMAN **65.73%** |

若跳过 stage-1，encoder 仍为 LAION 预训练，首轮 val 仅 ~73.8%。

### 4.3 脚本与产物

```bash
bash scripts/run_p1_train.sh
```

| 产物 | 路径 |
|------|------|
| P1 best（含 encoder） | `outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth` |
| GT 评测 | `p1_unfreeze4_noweight/eval_gt_roi/summary.json` |
| SegMAN 评测 | `p1_unfreeze4_noweight/eval_segman_roi/summary.json` |

**闸门**：GT ≥ T2 ✅；SegMAN +1.12 pt ✅。

---

## 5. P2 — 层次级联分类（⚠️ 部分通过）

### 5.1 架构

```text
Stage-1 路由：structure vs object
  ├─ structure → Stage-2 专头（door / wall / window）
  └─ object     → P1 11 类头（door/wall/window logit 置 -inf）
```

### 5.2 实施步骤

| 步骤 | 内容 | 状态 |
|------|------|------|
| P2-1 | 构建 `trans10k_roi_gt_hier` / `trans10k_roi_segman_hier` | ✅ |
| P2-2 | Stage-1 路由训练 | ⚠️ object recall 90.4%（闸门 92% 未达） |
| P2-3 | Stage-2 结构专头 | ✅ wall F1 81.1% |
| P2-4 | 级联 GT + SegMAN 评测 | ✅ |

### 5.3 结果

| 指标 | P1 | P2 级联 | Δ |
|------|-----|---------|---|
| GT-ROI Acc | 74.91% | **75.23%** | +0.32 pt |
| SegMAN-ROI Acc | 65.73% | **67.09%** | +1.36 pt |
| door F1 (GT) | 64.77% | 62.95% | −1.82 pt |

**决策**：不替换 deliver；P2 可作为 SegMAN demo 增强候选。脚本：`scripts/run_p2_1_build_hier_roi.sh` … `run_p2_4_eval_cascade.sh`。

---

## 6. P3 — Hard Mining + 增广（✅ 通过，现为正式 deliver）

### 6.1 目的

对 door/wall/window 等 hard 类过采样 + 轻量增广，从 P1 续训，目标 GT +1～2 pt。

### 6.2 实施流程

```text
P3-0  混淆 Top 错分对审计（P1 door↔wall 合计 434）
P3-1  构建 trans10k_roi_gt_p3（door/wall/window 2× 采样权重）
P3-2  ColorJitter + 旋转 + 同类 CutMix
P3-3  从 P1 best.pth 续训（lr 1e-6 / head 3e-5，patience 4）
P3-4  GT + SegMAN 评测
P3-5  闸门验收
```

```bash
bash scripts/run_p3_train.sh
```

### 6.3 训练曲线（best @ epoch 5）

| Epoch | val_acc | 说明 |
|-------|---------|------|
| 0 | 75.14% | |
| 5 | **76.91%** | **best** |
| 9 | 76.52% | early stop |

### 6.4 结果

| 指标 | P1 | **P3** | Δ |
|------|-----|--------|---|
| GT-ROI Acc | 74.91% | **76.91%** | **+2.00 pt** |
| SegMAN-ROI Acc | 65.73% | **67.49%** | +1.76 pt |
| door F1 (GT) | 64.77% | **66.61%** | +1.84 pt |
| wall F1 (GT) | 77.25% | **79.69%** | +2.44 pt |
| door↔wall 混淆 | 434 | **396** | −38 |

**闸门 PASS**（超 P1、door F1 ≥66%、SegMAN ≥65.5%）；未达 77% stretch。

### 6.5 升级为正式 deliver（2026-05-26）

应需求将 P3 设为正式交付：

```bash
bash scripts/promote_p3_deliver.sh
```

| 操作 | 路径 |
|------|------|
| 当前 deliver | `deliver_classifier_best.pth` ← P3 `best.pth` |
| T2 备份 | `deliver_classifier_t2_archived.pth` |
| manifest | `deliver_p3/deliver_manifest.json` |

---

## 7. P3+P2 级联 ablation（❌ 不采用）

**配置**：P2 Stage-1 + Stage-2 不变，object-head 换为 P3。

```bash
bash scripts/run_p3p2_eval_cascade.sh
```

| 方案 | GT-ROI | SegMAN |
|------|--------|--------|
| P3 单头 | **76.91%** | **67.49%** |
| P2 级联 | 75.23% | 67.09% |
| **P3+P2** | 75.27% | 67.18% |

**原因**：structure 类仍走 P2 Stage-2 专头，P3 在 door/wall 上的提升无法传递。**不采用级联**。

---

## 8. P4 — 域适配验证（❌ 未达 78%）

### 8.1 P4 快速验证（small）

```bash
bash scripts/run_p4_validate.sh
```

| 方法 | GT-ROI | vs P3 |
|------|--------|-------|
| WiSE-FT（α=0.95） | 76.94% | +0.03 pt |
| **Contrastive small（ep0）** | **77.04%** | **+0.13 pt** |
| 78% 闸门 | — | ❌ FAIL |

### 8.2 P4-full（方案 A）

全量 15746 ROI，lr 2e-7，patience=1：

```bash
bash scripts/run_p4_full.sh
```

| Epoch | val_acc |
|-------|---------|
| 0 | 76.91%（与 P3 持平） |
| 1 | 76.39% → early stop |

**结论**：全量 contrastive **未超 P4-small**；方案 A 终止。实验最高 GT 仍为 **P4-small 77.04%**（未作 deliver，ep0 峰值不稳定）。

---

## 9. 全阶段指标汇总

| 阶段 | GT-ROI Acc | SegMAN-ROI Acc | 备注 |
|------|------------|----------------|------|
| T2 deliver（已归档） | 74.88% | 64.61% | 首版正式交付 |
| P1 | 74.91% | 65.73% | encoder 持久化 |
| P2 级联 | 75.23% | 67.09% | demo 候选 |
| **P3 deliver（当前）** | **76.91%** | **67.49%** | **正式交付** |
| P3+P2 | 75.27% | 67.18% | 不采用 |
| P4-small（实验最高） | **77.04%** | 67.68% | 未 deliver |
| P4-full | 76.91% | 67.65% | FAIL |
| **课题目标** | **80.00%** | — | **未达标（−3.09 pt）** |

**相对 T2 deliver 提升（P3）**：

- GT-ROI：**+2.03 pt**
- SegMAN-ROI：**+2.88 pt**
- ΔAcc：−10.27 pt → **−9.42 pt**（全链路差距略缩小）

---

## 10. 为何未达 80% GT-ROI

1. **结构类混淆**：door↔wall 仍为 Top 错分对（P3 合计 396 次）。
2. **CLIP 表征上限**：ViT-B-16 轻量微调下，GT-ROI 约在 **77% 附近**触顶（P4-small ep0）。
3. **ROI 与分割约束**：window/shelf 等 weak 类 F1 低；SegMAN-ROI 距 GT 仍差 **9.42 pt**。
4. **已证无效路径**：class_weights、ViT-L 冻结、更紧 bbox、P0 弱类分割 finetune、P3+P2 级联、P4-full contrastive。

---

## 11. 当前正式交付规格

### 11.1 分类

| 项 | 值 |
|----|-----|
| Checkpoint | `outputs/openclip_classifier/deliver_classifier_best.pth` |
| 源文件 | `p3_p1_hardmining/best.pth` |
| 方法 | P1 encoder + P3 hard 2× + augment + unfreeze 4 blocks |
| GT-ROI | **76.91%**（3105 samples） |
| SegMAN-ROI | **67.49%**（3233 samples） |
| Manifest | `deliver_p3/deliver_manifest.json` |

### 11.2 分割

| 项 | 值 |
|----|-----|
| Checkpoint | `iter_6000.pth`（v2@6k） |
| mIoU | 81.80% |

### 11.3 回滚

如需恢复 T2 分类 deliver：

- 使用 `deliver_classifier_t2_archived.pth` 覆盖 `deliver_classifier_best.pth`
- 参考 `deliver_t2_best/deliver_manifest.json`

---

## 12. 方案 B — 拒识与双指标结题（已执行）

**执行日期**：2026-05-26  
**一键脚本**：`scripts/run_plan_b.sh`  
**产物目录**：`outputs/openclip_classifier/plan_b/`、`deliver_experiment_best/`

### 12.1 实施内容

| 步骤 | 内容 | 状态 | 脚本/产物 |
|------|------|------|-----------|
| B2 | Coverage–Accuracy 曲线（P3 deliver） | ✅ | `eval_coverage_accuracy.py` → `plan_b/coverage_*/` |
| B3 | 按类置信度拒识（door/wall 高 τ） | ✅ | `eval_reject_policy.py` + `configs/reject_thresholds_p3.json` |
| B4 | ROI 批量推理 + 拒识输出 | ✅ | `pipelines/classify_roi_with_reject.py` → `predictions_gt_val.json` |
| B5 | 结题摘要 + manifest | ✅ | `deliver_experiment_best/metrics_summary.md`、`manifest.json` |

### 12.2 实测结果

#### Coverage–Accuracy（B2）

| 评测集 | 全局 Top-1 | @60% coverage Acc | @70% coverage Acc |
|--------|-----------|-------------------|-------------------|
| GT-ROI | 76.91% | **89.16%** | **86.57%** |
| SegMAN-ROI | 67.49% | **80.67%** | 77.51% |

**闸门**：GT @60% coverage ≥78% → **PASS**  
**闸门**：GT @70% coverage ≥80% → **PASS**

#### 拒识策略（B3）

| 策略 | GT coverage | GT acc on accepted | SegMAN coverage | SegMAN acc |
|------|-------------|-------------------|-----------------|------------|
| 全局 τ=0.5 | 86.83% | 81.53% | 81.60% | 73.96% |
| **按类 τ** | **71.05%** | **86.08%** | **63.44%** | **80.01%** |

按类阈值见 `transgrasp/classification/configs/reject_thresholds_p3.json`（weak 类 door/wall/shelf 设更高 τ）。

### 12.3 组合验收（修订口径）

| 层级 | 指标 | 结果 | 门槛 |
|------|------|------|------|
| 分割 | mIoU | 81.80% | ≥80% ✅ |
| 分类上界 | GT-ROI Top-1 | 76.91% | ≥75% ✅ / 80% stretch ❌ |
| 部署 | SegMAN-ROI Top-1 | 67.49% | ≥65% ✅ |
| **系统** | **高置信子集 Acc** | **89.16% @60% cov** | **≥78% ✅** |
| **系统** | **按类拒识 Acc** | **86.08% @71% cov** | demo 安全 ✅ |

### 12.4 复现命令

```bash
docker exec segman_train bash -lc \
  'source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman && \
   cd /workspace/segman && bash scripts/run_plan_b.sh'
```

### 12.5 可选后续（B7）

P0 弱类分割 finetune 可并行提升 SegMAN-ROI 全局 Top-1（当前 67.49%）；不影响方案 B 结题结论。详见 `docs/E2E_性能分析与改进方案.md`（E1～E4）。

---

## 13. 关键文件与脚本索引

### 13.1 交付与 manifest

| 文件 | 说明 |
|------|------|
| `deliver_classifier_best.pth` | **当前 P3 deliver** |
| `deliver_classifier_t2_archived.pth` | T2 备份 |
| `deliver_p3/deliver_manifest.json` | P3 交付 manifest |
| `deliver_p3/eval_gt_roi/` | GT 评测报告 |
| `deliver_p3/eval_segman_roi/` | SegMAN 评测报告 |
| `plan_b/gate.json` | 方案 B 闸门汇总 |
| `deliver_experiment_best/metrics_summary.md` | 结题一页纸 |

### 13.2 各阶段 checkpoint

| 阶段 | 路径 |
|------|------|
| T2 | `t2_unfreeze2_noweight/best.pth` |
| P1 | `p1_unfreeze4_noweight/best.pth` |
| P2 Stage-1/2 | `p2_stage1_router/best.pth`，`p2_stage2_structure/best.pth` |
| P3 | `p3_p1_hardmining/best.pth` |
| P4-small | `p4_contrastive_small/best.pth` |

### 13.3 一键脚本

| 脚本 | 用途 |
|------|------|
| `scripts/run_p0_remaining.sh` | P0 剩余步骤 |
| `scripts/run_p1_train.sh` | P1 两阶段训练 |
| `scripts/run_p2_*` | P2 数据/训练/评测 |
| `scripts/run_p3_train.sh` | P3 全流程 |
| `scripts/run_p3p2_eval_cascade.sh` | P3+P2 级联评测 |
| `scripts/run_p4_validate.sh` | P4 快速验证 |
| `scripts/run_p4_full.sh` | P4-full |
| `scripts/promote_p3_deliver.sh` | T2→P3 deliver 升级 |
| `scripts/run_plan_b.sh` | 方案 B：coverage + 拒识 + 结题摘要 |
| `scripts/run_e2e_smoke.sh` | E2E 单图冒烟 |
| `scripts/run_e2e_eval_val.sh` | E2E val 子集评测 |
| `transgrasp/pipelines/segment_and_classify.py` | 原图→分割→分类 E2E |
| `docs/E2E_segment_and_classify_测试说明.md` | E2E 测试文档 |

### 13.4 Docker 环境

```bash
docker exec segman_train bash -lc \
  'source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman && \
   cd /workspace/segman && bash scripts/<script>.sh'
```

工作目录：`/workspace/segman`（挂载自仓库 `SegMAN/`）。

---

## 14. 修订记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-05-26 | v1.0 | 首版：P0～P4 完整历程、P3 deliver 升级、方案 B 建议 |
| 2026-05-26 | v1.1 | **方案 B 执行完成**：GT @60% cov **89.16%**；按类拒识 **86.08%**；结题摘要归档 |

---

## 15. 一句话总结

在 SegMAN v2@6k 分割与 OpenCLIP ViT-B-16 基线上，经 **P0～P4 系统优化**，分类 GT-ROI 从 T2 **74.88%** 提升至 P3 **76.91%**（**正式 deliver**），SegMAN-ROI 从 **64.61%** 提升至 **67.49%**；**80% 全局 Top-1 未达标**。经 **方案 B**，高置信子集（coverage 60%）GT Acc **89.16%**，按类拒识有效 Acc **86.08%**，**双指标结题闸门 PASS**。实验最高 GT 为 P4-small **77.04%**。
