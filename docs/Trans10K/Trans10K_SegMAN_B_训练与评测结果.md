# Trans10K-v2 · SegMAN-B 训练与验证集评测结果

> 记录时间：2026-05-18  
> 工作目录：`segmentation/outputs/trans10k_segman_b/`  
> 配置：`local_configs/segman_trans/segman_b_trans10k.py`

---

## 1. 实验设置

| 项目 | 内容 |
|------|------|
| 模型 | SegMAN-B（`SegMANEncoder_b` + `SegMANDecoder`） |
| 预训练 | `pretrained/SegMAN_Encoder_b.pth.tar` |
| 数据集 | Trans10K-v2 → MMSeg `data/trans10k` |
| 类别数 | 12（background + 11 类透明物体） |
| 训练迭代 | **80,000**（`runner.max_iters=80000`） |
| Batch | `samples_per_gpu=2`，`workers_per_gpu=2` |
| 优化器 | AdamW，`lr=6e-5`，poly schedule |
| Checkpoint 间隔 | 每 4000 iter 保存 |
| 训练中验证 | 续训阶段使用 `--no-validate`（避免验证 OOM）；iter 8000 时曾完整验证 |

**课题分工（分割 vs 分类）**

- **SegMAN（本文）**：透明物体 **语义分割**（像素级 mask），供下游 ROI 裁剪。  
- **细分类**：由路线 C 的 **Grounded-SAM 特征 + TransFine** 完成，**不以 SegMAN 12 类 logits 作为最终分类结果**。

---

## 2. 训练过程摘要

| 阶段 | 说明 |
|------|------|
| 冒烟 | `outputs/trans10k_smoke`，约 20 iter |
| Debug | `outputs/trans10k_debug`，2000 iter，val mIoU ≈ **23.42%**（@2000） |
| 正式 1 | iter 0→16000，iter 8000 验证 mIoU ≈ **63.46%** |
| 中断 | iter 16000 验证时进程 **Killed**（内存 OOM） |
| 正式 2 | 自 `iter_16000.pth` 续训，`--no-validate`，至 **iter 80000** 完成 |

**训练结束日志（iter 80000）**

| 指标 | 数值 |
|------|------|
| `decode.loss_ce` | 0.0643 |
| `decode.acc_seg`（训练集像素准确率） | 97.62% |
| 学习率 | 7.500e-10 |

> 注：`acc_seg` 为训练集统计，**不等于** 验证集 mIoU。

---

## 3. 验证集评测（test.py）

**命令（本仓库 test.py 需使用 `--checkpoint`）**

```bash
cd /workspace/segman/segmentation
conda activate segman

python tools/test.py local_configs/segman_trans/segman_b_trans10k.py \
  --checkpoint outputs/trans10k_segman_b/iter_80000.pth \
  --eval mIoU
```

**可选可视化**

```bash
python tools/test.py local_configs/segman_trans/segman_b_trans10k.py \
  --checkpoint outputs/trans10k_segman_b/iter_80000.pth \
  --eval mIoU \
  --show-dir outputs/trans10k_segman_b/vis_val
```

**评测设置**

| 项目 | 内容 |
|------|------|
| 权重 | `iter_80000.pth` |
| 验证集规模 | 1000 张 |
| 耗时（参考） | 约 458 s（~2.2 task/s） |

---

## 4. 验证集总体指标

| 指标 | 数值 (%) |
|------|----------|
| **mIoU** | **80.71** |
| aAcc | 96.07 |
| mAcc | 88.14 |

---

## 5. 验证集各类 IoU / Acc

| ID | 类别 | IoU (%) | Acc (%) |
|----|------|---------|---------|
| 0 | background | 96.71 | 98.22 |
| 1 | box | 71.47 | 80.57 |
| 2 | bottle | 87.77 | 89.93 |
| 3 | window | 66.62 | 90.77 |
| 4 | eyeglass | 92.85 | 95.92 |
| 5 | freezer | 73.90 | 77.31 |
| 6 | jar_kettle | 84.04 | 88.32 |
| 7 | door | 75.04 | 83.98 |
| 8 | cup | 90.91 | 96.55 |
| 9 | wall | 82.72 | 91.99 |
| 10 | bowl | 78.91 | 85.51 |
| 11 | shelf | 67.61 | 78.56 |

### 5.1 简要分析

**IoU 较高（≥ 85%）**  
background、bottle、eyeglass、cup

**IoU 中等（75%～85%）**  
jar_kettle、wall、bowl、door、freezer

**IoU 偏低（< 75%，后续改进重点）**  
window（66.62）、shelf（67.61）、box（71.47）

**现象**  
部分类别 Acc 明显高于 IoU（如 window：Acc 90.77 vs IoU 66.62），常见原因为类别大体判对但 **边界不准或区域漏分**，与透明物体边缘难分割一致。

---

## 6. 不同阶段 mIoU 对比

| 阶段 | Checkpoint | test 命令 | val mIoU (%) | aAcc (%) | mAcc (%) |
|------|------------|-----------|--------------|----------|----------|
| Debug | `trans10k_debug`，iter 2000 | — | ≈ 23.42 | — | — |
| 正式 @8k | `best_mIoU_iter_8000.pth` | 已测（2026-05-18） | **63.46** | 92.57 | 77.62 |
| **正式 @80k** | **`iter_80000.pth`** | 已测（2026-05-18） | **80.71** | 96.07 | 88.14 |

**相对 `best_mIoU_iter_8000.pth`，`iter_80000.pth`：mIoU +17.25 个百分点，aAcc +3.50，mAcc +10.52。**

**定稿权重：仅用 `iter_80000.pth`。** `best_mIoU_iter_8000.pth` 仅作训练中期对照，不用于部署或下游。

### 6.1 8k best vs 80k：各类 IoU 对比（百分点）

| 类别 | best @8k | iter @80k | Δ (80k−8k) |
|------|----------|-----------|------------|
| background | 93.99 | 96.71 | +2.72 |
| box | 51.88 | 71.47 | +19.59 |
| bottle | 72.37 | 87.77 | +15.40 |
| window | 74.65 | 66.62 | **−8.03** |
| eyeglass | 75.72 | 92.85 | +17.13 |
| freezer | 25.31 | 73.90 | +48.59 |
| jar_kettle | 64.50 | 84.04 | +19.54 |
| door | 49.97 | 75.04 | +25.07 |
| cup | 84.95 | 90.91 | +5.96 |
| wall | 73.22 | 82.72 | +9.50 |
| bowl | 70.28 | 78.91 | +8.63 |
| shelf | 24.64 | 67.61 | +42.97 |

**说明**

- 80k 在 **11/12 类**（含物体类）IoU 高于 8k；**window** 在 8k 时 IoU 虚高（74.65），80k 反而降至 66.62，可能与长训后边界更严、评估更保守有关，需结合 `vis_val` 目视。  
- 8k 时 **freezer、shelf、door、box** 极差（IoU 25～52），80k 大幅提升，说明 **必须训满 80k**，不能停在 `best_mIoU`。

### 6.2 `best_mIoU_iter_8000.pth` 完整 per-class（2026-05-18 test）

| 类别 | IoU (%) | Acc (%) |
|------|---------|---------|
| background | 93.99 | 97.56 |
| box | 51.88 | 56.70 |
| bottle | 72.37 | 76.29 |
| window | 74.65 | 77.05 |
| eyeglass | 75.72 | 96.31 |
| freezer | 25.31 | 90.80 |
| jar_kettle | 64.50 | 86.10 |
| door | 49.97 | 65.00 |
| cup | 84.95 | 92.70 |
| wall | 73.22 | 82.46 |
| bowl | 70.28 | 82.95 |
| shelf | 24.64 | 27.55 |

评测耗时约 286 s（~3.5 task/s）。

---

## 7. 产出文件路径

| 文件 | 说明 |
|------|------|
| `segmentation/outputs/trans10k_segman_b/iter_80000.pth` | **推荐部署 / 下游使用的最终权重** |
| `segmentation/outputs/trans10k_segman_b/latest.pth` | 通常指向最新 iter |
| `segmentation/outputs/trans10k_segman_b/best_mIoU_iter_8000.pth` | 训练中验证最优；**test mIoU 63.46%**，低于 80k，仅作对照 |
| `segmentation/outputs/trans10k_segman_b/iter_16000.pth` 等 | 中间 checkpoint（每 4000 iter） |
| `segmentation/outputs/trans10k_segman_b/vis_val/` | 验证集可视化（`--show-dir`） |
| `segmentation/outputs/trans10k_segman_b/*.log` | 训练日志 |

Docker 内绝对路径示例：

```text
/workspace/segman/segmentation/outputs/trans10k_segman_b/iter_80000.pth
```

---

## 8. 结论与后续工作

**结论**

- SegMAN-B 在 Trans10K-v2 验证集上达到 **mIoU 80.71%**，基线分割训练 **已完成**。  
- 定稿权重：**`iter_80000.pth`**。  
- 系统集成时使用 **分割 mask**（多类预测可将 `label > 0` 合并为透明前景），细分类进入路线 C。

**建议后续**

| 路线 | 内容 |
|------|------|
| B | 实现 LASS + MMSCopE，目标 mIoU > 80.71，重点改善 window / shelf / box → **《路线B_LASS_MMSCopE_实施清单.md》** |
| C | mask → ROI 数据集 → Grounded-SAM 特征 → TransFine 细分类 → 抓取/UI |

详见：《透明物体分割_SegMAN优化设计说明书.md》《项目实施步骤指南.md》《Trans10K训练快速开始.md》第 6 步。

---

## 9. 论文/报告用一行摘要（可复制）

> SegMAN-B 在 Trans10K-v2 上训练 80k iterations，验证集 mIoU **80.71%**（aAcc 96.07%，mAcc 88.14%），权重见 `iter_80000.pth`。
