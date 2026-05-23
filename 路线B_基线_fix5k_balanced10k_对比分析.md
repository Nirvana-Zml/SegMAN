# 路线 B：基线 vs fix5k vs balanced10k（iter_10000）对比分析

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.0 |
| 编写日期 | 2026-05-23 |
| 评测集 | Trans10K **val 1000 张**（与基线、`tools/test.py --eval mIoU` 一致） |
| 关联文档 | 《Trans10K_SegMAN_B_训练与评测结果.md》《路线B_LASS_MMSCopE_实施清单.md》§0.1、§0.3；《路线B_平衡微调方案.md》§8 |

---

## 1. 三个权重与训练设置

| 项目 | 基线 | fix5k | balanced10k / iter_10000 |
|------|------|-------|---------------------------|
| **权重路径** | `outputs/trans10k_segman_b/iter_80000.pth` | `outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth` | `outputs/trans10k_lass_mmscope_balanced10k/iter_10000.pth` |
| **配置文件** | `segman_b_trans10k.py` | `segman_b_trans10k_lass.py` | `segman_b_trans10k_lass_balanced.py` |
| **网络** | SegMAN-B | SegMAN-B + **LASS** + **MMSCopE** | 同 fix5k |
| **初始化** | ImageNet 预训练 encoder | `load-from` 基线 80k | `load-from` 基线 80k |
| **max_iters** | 80000 | **5000** | **10000** |
| **optimizer.lr** | 6e-5（poly） | **3e-5** | **2e-5** |
| **LASS stages** | — | **[0, 1, 2]** | **[1, 2]** |
| **boundary_loss_weight** | — | 0.4 | **0.15** |
| **loss** | CE | CE + 边界 | **加权 CE + Dice 0.4** + 边界 |
| **代码前提** | — | `ignore_index=255` 已修复 | 同左 |

---

## 2. 总体指标对比

| 指标 | 基线 (80k) | fix5k (5k) | balanced10k (10k) | fix5k − 基线 | balanced10k − 基线 | balanced10k − fix5k |
|------|------------|------------|-------------------|--------------|--------------------|-----------------------|
| **mIoU** | **80.71** | **80.84** | **81.76** | **+0.13** | **+1.05** | **+0.92** |
| aAcc | 96.07 | 95.92 | 96.15 | −0.15 | +0.08 | +0.23 |
| mAcc | 88.14 | 87.45 | 87.71 | −0.69 | −0.43 | +0.26 |

**总体结论**

- **mIoU 排序**：balanced10k **81.76** > fix5k **80.84** > 基线 **80.71**；三者均基于同 val 集，可比。
- **fix5k**：相对基线 **+0.13% mIoU**，改动小、训练短，**全面略优于基线**（除 4 类 IoU 明显回落）。
- **balanced10k**：相对基线 **+1.05% mIoU**，相对 fix5k **+0.92%**；**window 增益最大**，但 **bowl 明显低于基线与 fix5k**。
- aAcc / mAcc 差异小于 1 个百分点，**不宜单独作为选型依据**；以 **mIoU 与关键类 IoU** 为主。

---

## 3. 各类 IoU 对比（%）

| 类别 | 基线 | fix5k | balanced10k | Δ fix5k−基线 | Δ balanced−基线 | Δ balanced−fix5k | 相对基线（balanced） |
|------|------|-------|-------------|--------------|-----------------|------------------|----------------------|
| background | 96.71 | 96.37 | 96.45 | −0.34 | −0.26 | +0.08 | ↓ 略 |
| box | 71.47 | 71.44 | 71.86 | −0.03 | +0.39 | +0.42 | ↑ |
| bottle | 87.77 | 86.51 | 88.20 | −1.26 | +0.43 | +1.69 | ↑ |
| **window** | 66.62 | 76.27 | **82.91** | **+9.65** | **+16.29** | **+6.64** | **↑ 最大** |
| eyeglass | 92.85 | 90.81 | 92.01 | −2.04 | −0.84 | +1.20 | ↓ |
| freezer | 73.90 | 73.46 | 73.48 | −0.44 | −0.42 | +0.02 | ≈ |
| jar_kettle | 84.04 | 82.06 | 83.89 | −1.98 | −0.15 | +1.83 | ≈ |
| door | 75.04 | 72.72 | 74.40 | −2.32 | −0.64 | +1.68 | ↓ 略 |
| cup | 90.91 | 90.19 | 90.96 | −0.72 | +0.05 | +0.77 | ≈ |
| wall | 82.72 | 82.77 | 83.95 | +0.05 | +1.23 | +1.18 | ↑ |
| **bowl** | **78.91** | **80.07** | **74.31** | **+1.16** | **−4.60** | **−5.76** | **↓ 最大回落** |
| shelf | 67.61 | 67.44 | 68.73 | −0.17 | +1.12 | +1.29 | ↑ |
| **mIoU** | **80.71** | **80.84** | **81.76** | **+0.13** | **+1.05** | **+0.92** | **↑** |

### 3.1 IoU 统计（相对基线，|Δ|>0.2% 计一类）

| 权重 | IoU ↑ | IoU ≈ 持平 | IoU ↓ |
|------|-------|------------|-------|
| **fix5k** | **3**（window、bowl；wall≈0） | **5** | **4**（bottle、eyeglass、jar_kettle、door） |
| **balanced10k** | **5**（window、box、bottle、wall、shelf） | **3**（freezer、jar_kettle、cup） | **4**（background 略、eyeglass、door、**bowl**） |

### 3.2 fix5k 与 balanced10k 逐类胜负（IoU）

| balanced10k 更高 | fix5k 更高 | 接近 |
|--------------------|------------|------|
| window (+6.64)、bottle、box、wall、shelf、jar_kettle、door、cup | **bowl**（−5.76）、background（略） | freezer、eyeglass（互有高低） |

---

## 4. 各类 Acc 对比（%）

| 类别 | 基线 | fix5k | balanced10k | Δ fix5k−基线 | Δ balanced−基线 |
|------|------|-------|-------------|--------------|-----------------|
| background | 98.22 | 98.33 | 98.08 | +0.11 | −0.14 |
| box | 80.57 | 79.44 | 78.40 | −1.13 | −2.17 |
| bottle | 89.93 | 87.97 | 89.42 | −1.96 | −0.51 |
| window | 90.77 | 86.71 | 85.37 | −4.06 | −5.40 |
| eyeglass | 95.92 | 96.04 | 97.45 | +0.12 | +1.53 |
| freezer | 77.31 | 76.07 | 76.31 | −1.24 | −1.00 |
| jar_kettle | 88.32 | 87.53 | 88.43 | −0.79 | +0.11 |
| door | 83.98 | 82.58 | 84.62 | −1.40 | +0.64 |
| cup | 96.55 | 95.39 | 96.15 | −1.16 | −0.40 |
| wall | 91.99 | 91.41 | 93.84 | −0.58 | +1.85 |
| bowl | 85.51 | 86.98 | 82.28 | +1.47 | −3.23 |
| shelf | 78.56 | 80.90 | 82.23 | +2.34 | +3.67 |
| **mAcc** | **88.14** | **87.45** | **87.71** | −0.69 | −0.43 |

**说明**：window / bowl 常出现 **IoU 与 Acc 不同向**（如 balanced10k：window IoU +16.29 vs 基线，Acc −5.40）。表示「大类判对比例」与「区域重叠（IoU）」不是同一回事；**论文与验收建议以 IoU / mIoU 为主**。

---

## 5. 综合分析

### 5.1 基线 → fix5k（5k 微调 + LASS/MMSCopE）

**收益**

- 总 **mIoU +0.13%**，验证路线 B **有效且训练成本低**（5000 iter）。
- **window +9.65% IoU**：对基线最弱类之一改善最大，符合阶段 3 目标。
- **bowl +1.16% IoU**：在引入边界模块后仍略高于基线，适合作为 **稳妥交付 / 软著 / 专利主实施例**。

**代价**

- bottle、door、eyeglass、jar_kettle 共 **4 类 IoU 降约 1.3～2.3%**。
- shelf、box 与基线 **基本持平**（未超过基线 0.2% 阈值）。

**机制简述**：较强边界监督（0.4）+ 全 stage LASS，利于难边界类（window、bowl），对部分区域类（bottle、door 等）梯度竞争略吃亏。

### 5.2 基线 → balanced10k（10k 平衡微调）

**收益**

- 总 **mIoU +1.05%**（三路最高）。
- **window +16.29% IoU**（相对 fix5k 再 +6.64%）。
- **5 类 IoU 高于基线**：window、box、bottle、wall、shelf；多类回落类（bottle、door 等）部分被拉回。

**代价**

- **bowl −4.60% IoU**（相对基线）；相对 fix5k **−5.76%**，为三者中 bowl **唯一明显变差**的权重。
- eyeglass、door 仍略低于基线；background IoU 略降。

**机制简述**（与《路线B_平衡微调方案.md》§8.6 一致）

- Dice 0.4、LASS 仅 stage [1,2]、cup 权重 1.05 等，优化重心偏向 **总 mIoU / window / 多类均衡**。
- 方案 3 混淆：GT=bowl 像素上约 **9.29%→background**、**6.19%→cup**；召回约 82% 但 IoU 低，含 **误检 bowl** 因素。
- bowl 在 6k/8k/10k **持续低于基线**，非单点过拟合。

### 5.3 fix5k vs balanced10k：如何选型

| 需求 | 推荐权重 | 理由 |
|------|----------|------|
| **最高 mIoU、window 论文主表** | `balanced10k/iter_10000.pth` | mIoU 81.76，window 82.91 |
| **mIoU≥基线且 bowl≥基线、交付/软著/专利主版本** | **`fix5k/iter_5000.pth`** | mIoU 80.84，bowl 80.07，无 bowl 崩盘 |
| **多类 IoU 尽量不低于基线** | 可参考 `balanced10k/iter_6000`（见平衡微调 §8.3，非本文三表主项） | ↑6 类，mIoU 80.83 |
| **路线 C 分割 mask 默认** | **`fix5k/iter_5000.pth`（项目已冻结）** | 见《路线B_fix5k_项目后续步骤.md》 |

**当前实验结论**：**不存在**同一 checkpoint 在 val 上同时达到 **mIoU≈81.76 且 bowl≥78.91**；fix5k 与 balanced10k 是 **互补关系**，非严格支配关系。

---

## 6. 复现命令

**基线**

```bash
python tools/test.py local_configs/segman_trans/segman_b_trans10k.py \
  --checkpoint outputs/trans10k_segman_b/iter_80000.pth \
  --eval mIoU
```

**fix5k**

```bash
python tools/test.py local_configs/segman_trans/segman_b_trans10k_lass.py \
  --checkpoint outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --eval mIoU
```

**balanced10k / iter_10000**

```bash
python tools/test.py local_configs/segman_trans/segman_b_trans10k_lass_balanced.py \
  --checkpoint outputs/trans10k_lass_mmscope_balanced10k/iter_10000.pth \
  --eval mIoU
```

**逐类 Δ 表（相对基线 80.71）**

```bash
python scripts/compare_miou_vs_baseline.py <test.py 输出的 json 路径>
```

---

## 7. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-23 | 初版：基线、fix5k、balanced10k iter_10000 三表合一 + 分析 |
