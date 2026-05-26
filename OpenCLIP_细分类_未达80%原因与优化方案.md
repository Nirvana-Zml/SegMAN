# OpenCLIP 透明物体 ROI 细分类 — 未达 80% 原因分析与优化方案


| 项目              | 内容                                                        |
| --------------- | --------------------------------------------------------- |
| 文档版本            | v2.7                                                      |
| 编写日期            | 2026-05-26                                                |
| 关联文档            | 《OpenCLIP_细分类训练与优化指南.md》《路线C_细分类与抓取实施步骤.md》**《OpenCLIP_细分类_完整优化历程与交付说明.md》**（P0～P4 全过程） |
| 当前交付 checkpoint | `outputs/openclip_classifier/deliver_classifier_best.pth`（**P3**，2026-05-26 升级） |
| 验收目标（课题）        | GT-ROI val **Top-1 Acc ≥ 80%**                            |
| 当前交付指标          | GT-ROI **76.91%**（P3）/ SegMAN-ROI **67.49%**（P3 + v2@6k） |
| 实验最高（非 deliver） | GT-ROI **77.04%**（P4 contrastive small ep0）              |
| 上一版正式交付（已归档）    | T2 GT **74.88%** / SegMAN **64.61%** → `deliver_classifier_t2_archived.pth` |
| 差距（vs 80% GT）   | **−3.09 pt**（当前 P3 deliver）                                  |


---

## 1. 摘要

在 SegMAN v2@6k（mIoU **81.80%**）分割基础上，经 T1 → T2 → P1 → P2 → P3 → P4，**正式交付已升级为 P3 单头：GT-ROI 76.91% / SegMAN-ROI 67.49%**（较原 T2 deliver **+2.03 / +2.88 pt**）。实验最高 GT **77.04%**（P4-small，未作 deliver）。**80% GT 硬指标仍未达标**（差 **3.09 pt**）。

**核心结论**：

1. **80% 未达标的主因**是 **任务固有难度 + 类间结构混淆 + 表征/数据上限** 叠加。
2. **约 443 个 val ROI 被分错**（3105×(1−0.7488)≈780，其中大量为 door/wall/window 系统性混淆，非随机噪声）。
3. **SegMAN-ROI 仅 64.61%**，说明即使分类在 GT 上冲到 80%，**全链路距 80% 仍差 ~15 pt**；分割与分类需分开优化。
4. §9 已验证的 **ViT-L 冻结 linear、缩小 bbox、§8 式调参** 均无法补齐缺口；继续同路线微调 **性价比极低**。

---

## 2. 当前结果快照

### 2.1 正式归档指标（P3 deliver，`deliver_classifier_best.pth`）

> **2026-05-26 升级**：自 T2 替换为 P3。T2 备份见 `deliver_classifier_t2_archived.pth`；manifest 见 `deliver_p3/deliver_manifest.json`。


| 评测集                | 样本数   | Top-1 Acc     | macro F1 | 说明     |
| ------------------ | ----- | ------------- | -------- | ------ |
| **GT-ROI val**     | 3,105 | **76.91%**    | 73.52%   | 分类理论上界 |
| **SegMAN-ROI val** | 3,233 | **67.49%**    | 61.45%   | 部署真实精度 |
| **ΔAcc**           | —     | **−9.42 pt**  | —        | 分割传递损失 |

训练：自 P1 `best.pth` 续训；`trans10k_roi_gt_p3` hard 2× + `--aug p3`；unfreeze **4** block；best @ **epoch 5**。

**上一版 T2 deliver（已归档）**：GT **74.88%** / SegMAN **64.61%** / ΔAcc **−10.27 pt**。

### 2.2 实验历程与 Acc 变化


| 阶段       | 方法                          | GT-ROI Acc | vs 上一最佳      | 结论       |
| -------- | --------------------------- | ---------- | ------------ | -------- |
| T1       | 冻结 ViT-B + linear + weights | 70.18%     | —            | baseline |
| §8       | 无 class_weights             | 72.62%     | +2.44 pt     | 有效       |
| **T2**   | §8 + 解冻末 2 block            | **74.88%** | **+2.26 pt** | **当前最优** |
| ViT-L-14 | 冻结 L-14 + linear            | 72.37%     | −2.51 pt     | 无效       |
| pad10    | bbox-pad 0.10 重训            | 70.82%     | −4.06 pt     | 有害       |


**观察**：每轮有效手段约 **+2～2.5 pt**；T2 后再走「同范式」路径（更大 backbone、更紧 ROI）**均为负增益**。

### 2.3 每类 F1（T2，GT-ROI）— 强弱分化


| 类          | support | F1        | 状态                      |
| ---------- | ------- | --------- | ----------------------- |
| eyeglass   | 92      | **92.7%** | 强                       |
| cup        | 366     | **91.7%** | 强                       |
| wall       | 1290    | **77.5%** | 中强（样本多，错分影响大）           |
| bottle     | 223     | 78.4%     | 中强                      |
| bowl       | 38      | 78.3%     | 中（样本少）                  |
| box        | 88      | 69.5%     | 弱                       |
| jar_kettle | 133     | 71.7%     | 中                       |
| door       | 663     | **63.5%** | **弱（高 support，拖累 Acc）** |
| window     | 130     | **51.2%** | **弱**                   |
| shelf      | 62      | **51.0%** | **弱**                   |
| freezer    | 20      | 68.6%     | 样本极少                    |


**粗算**：若 door/wall/window/shelf 四类 F1 平均从 ~55% 提到 ~75%，整体 Acc 约可 **+3～5 pt** — 这恰是冲 80% 需要补的缺口，但这四类正是 **结构混淆 + 分割弱项** 集中区。

---

## 3. 未达 80% 的根因分析

### 3.1 根因分层模型

```text
                    ┌─────────────────────────────────────┐
                    │  验收：GT-ROI Top-1 ≥ 80%           │
                    │  当前：74.88%  差距 5.12 pt         │
                    └─────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
  【任务与数据】                 【分类模型】                  【分割与 ROI】
  11 类透明物体互像            表征与决策边界不足            mask/裁框引入噪声
  door/wall/window 难辨        线性头→T2 仍不够              ΔAcc −10.27 pt
        │                             │                             │
        └─────────────────────────────┴─────────────────────────────┘
                                      │
                              系统性混淆（非随机）
                         door↔wall 各 200+ 次/val
```

### 3.2 原因一：类间语义与视觉高度重叠（任务难）

Trans10K 11 类前景中，多组类别在透明/反光条件下 **CLIP 预训练特征难以线性分离**：


| 易混组                                  | 表现      | 证据                                         |
| ------------------------------------ | ------- | ------------------------------------------ |
| **door ↔ wall**                      | 双向最大混淆  | GT val：door→wall **226**，wall→door **217** |
| **window ↔ wall**                    | 立面类互吸   | window→wall **67**                         |
| **bottle / cup / jar_kettle / bowl** | 容器互混    | jar→cup 17，bottle→cup 12                   |
| **shelf → wall/door**                | 弱类被吸到大类 | shelf→wall 21；SegMAN 上 shelf→wall **64**   |


这是 **决策边界问题**，不是简单「多训几个 epoch」能消除；T2 仅将 door F1 从 ~57% 提到 **63.5%**，window/shelf 仍 ~51%。

### 3.3 原因二：表征与模型容量瓶颈


| 手段                   | 参数量级    | 结果          | 说明                       |
| -------------------- | ------- | ----------- | ------------------------ |
| 冻结 ViT-B + linear    | 头 ~5.6K | 72.62% 封顶   | §8 调 lr/MLP/smoothing 无效 |
| T2 解冻末 2 block       | +ViT 尾部 | **74.88%**  | 唯一显著 +2 pt 手段            |
| ViT-L-14 冻结 + linear | 768 维特征 | 72.37%      | **低于** ViT-B T2          |
| class_weights        | —       | 72.62→70.18 | **有害**，放大噪声类             |


**解读**：

- LAION 预训练 CLIP 对 **透明域细粒度 11 类** 的零样本/线性 probe 天花板约在 **73%～75%**（T1+§8+T2 已验证）。
- 更大 backbone **未带来更好线性可分性**（ViT-L 反而更差），说明瓶颈在 **域适配与类间结构**，不在特征维度。
- T2 后 val 在 epoch 4 达峰、epoch 9 早停 → **继续加深微调有过拟合风险**，marginal gain 有限。

### 3.4 原因三：ROI 与标注方式的上限


| 因素               | 影响                                                           |
| ---------------- | ------------------------------------------------------------ |
| 语义 mask 连通域 bbox | 常含背景、邻物、反光；weak 类尤其严重                                        |
| bbox-pad 0.15    | 相对 0.10 **更好**（pad10 实验 Acc −4 pt）→ 上下文对 door/wall 分类 **必要** |
| 实例数不变            | pad10 与 pad0.15 均为 15746/3105 → 优化空间不在「多裁 ROI」               |
| GT mask 理想化      | GT-ROI 74.88% 仍 <80% → **即使完美 mask，分类头仍不足**                  |


### 3.5 原因四：分割误差对全链路的硬约束（部署向）

**C-4 归因（T2，GT F1 vs SegMAN F1）**：


| 类        | GT F1 | SegMAN F1 | Δ            | 瓶颈    |
| -------- | ----- | --------- | ------------ | ----- |
| freezer  | 68.6% | 34.6%     | **−33.9 pt** | 分割    |
| shelf    | 51.0% | 24.7%     | **−26.3 pt** | 分割    |
| box      | 69.5% | 46.0%     | **−23.5 pt** | 分割    |
| bowl     | 78.3% | 54.8%     | **−23.5 pt** | 分割    |
| door     | 63.5% | 46.6%     | **−16.9 pt** | 分割+分类 |
| eyeglass | 92.7% | 92.0%     | −0.8 pt      | 已较好   |


SegMAN mIoU **81.8%** 但 **per-class IoU 弱项**（shelf/window/door）直接压低 SegMAN-ROI；SegMAN 上 wall→door **325** 次，远超 GT 上同类混淆。

**结论**：**80% GT-ROI** 与 **80% SegMAN-ROI** 是不同难度；后者在当前分割下 **极难** 单靠分类达成。

### 3.6 原因五：已尝试路径的「收益递减」

§9 阶段 A/B/C 形成完整证据链：

- **A（T2）**：+2.26 pt ✅  
- **B（ViT-L）**：−2.51 pt ❌  
- **C（pad10）**：−4.06 pt ❌  
- **§8（lr/MLP/weights）**：仅「去 weights」+2.44 pt，其余无效 ❌

说明 **当前范式内的低成本优化已耗尽**；缺口的 ~5 pt 需要 **换手段**（见 §5）。

---

## 4. 误差量化：5.12 pt 从哪里来？

val 共 **3,105** ROI，Acc **74.88%** → 约 **781 个错误**；若 Acc **80%** → 约 **621 个错误**，需 **少错 ~160 个**（**5.12% × 3105 ≈ 159**）。

**按混淆对粗估可优化空间**（T2 GT-ROI）：


| 混淆类型              | 约占总错分    | 可优化性         |
| ----------------- | -------- | ------------ |
| door ↔ wall       | ~443 次双向 | 中（需结构/上下文特征） |
| window → wall     | ~67      | 中低           |
| shelf → wall/door | ~21+     | 低（样本少+分割差）   |
| 容器类互混             | ~50      | 中            |
| 其余                | 分散       | 难            |


即使 **door/wall 混淆减半**（约 −220 错分中的有效 TP），理论 Acc 提升约 **+3～4 pt** → 仍可能 **略低于 80%**；要稳定 ≥80% 需 **多类同时改善 + 少引入新错分**。

---

## 5. 优化方案（按优先级）

### 5.1 总览


| 优先级    | 方向                    | 目标指标            | 预期增益        | 难度  | 建议                  |
| ------ | --------------------- | --------------- | ----------- | --- | ------------------- |
| **P0** | 分割弱类专项提升              | SegMAN-ROI、ΔAcc | +3～8 pt（部署） | 高   | **全链路必做**           |
| **P1** | T2 加深 + 完整 checkpoint | GT-ROI          | +1～3 pt     | 中   | ✅ **已完成**（74.91%）   |
| **P2** | 层次/分组分类               | GT-ROI weak 类   | +2～4 pt（实测 +0.32 pt） | 中   | ✅ **已完成**（GT 75.23% / SegMAN 67.09%） |
| **P3** | 数据增强与 hard mining     | GT-ROI          | +1～2 pt（实测 **+2.00 pt**） | 中   | ✅ **已完成**（GT **76.91%** / SegMAN **67.49%**） |
| **P4** | 多模态/专用预训练             | GT-ROI          | +3～6 pt     | 很高  | ❌ **P4-full FAIL**；实验最高 **P4-small 77.04%** |
| **P5** | 集成与拒识策略               | 有效 Acc          | 子集 +5～10 pt | 低   | 交付/demo 用           |


以下 **不做**：ViT-L 冻结 linear、class_weights、bbox-pad<0.15、§8 式 lr  sweep。

---

### 5.2 P0 — 分割弱类专项（提升 SegMAN-ROI 与全链路）

**总目的**：分类 C-4 已证 **shelf / box / freezer / bowl / door** 的 GT→SegMAN F1 落差达 **17～34 pt**；在 **不改分类 checkpoint** 的前提下，通过改善 SegMAN 预测 mask → 重导 SegMAN-ROI → 重评 `deliver_classifier_best.pth`，提升 **SegMAN-ROI Acc** 与 **ΔAcc**。

**前置（固定不变）**：

| 组件 | 路径 | 说明 |
|------|------|------|
| 当前分割 | `segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth` | mIoU **81.80%**，**勿删** |
| 当前分类 | `outputs/openclip_classifier/deliver_classifier_best.pth` | T2 **74.88% / 64.61%** |
| ROI 默认 | `bbox-pad 0.15`，`min-area 64` | §9C 已证不宜改 pad |

**弱类对齐表（v2@6k 分割 IoU vs T2 分类 F1 落差）**：

| 类 | v2@6k IoU | T2 GT F1 | T2 SegMAN F1 | Seg−GT Δ | P0 优先级 |
|----|-----------|----------|--------------|----------|-----------|
| shelf | 67.73% | 51.0% | 24.7% | **−26.3 pt** | **P0 最高** |
| freezer | 73.39% | 68.6% | 34.6% | **−33.9 pt** | **P0 最高** |
| box | 72.97% | 69.5% | 46.0% | **−23.5 pt** | 高 |
| bowl | 80.70% | 78.3% | 54.8% | **−23.5 pt** | 高 |
| door | 73.34% | 63.5% | 46.6% | **−16.9 pt** | 高（+ wall 混淆） |
| window | 77.16% | 51.2% | 46.4% | −4.8 pt | 中（像素 IoU 不低，ROI/分类难） |

> **注意**：window 说明 **像素 IoU 高 ≠ ROI 分类好**；P0 仍需做 **ROI 级** 对齐（步骤 P0-0），不能只看 mIoU。

**P0 总流程**：

```text
P0-0  基线审计（per-class IoU + ROI 级 GT/SegMAN 对比）
  ↓
P0-1  分割弱类 finetune（从 iter_6000 短程微调，新 config）
  ↓
P0-2   导出新 pred mask → 重导 trans10k_roi_segman → 重评分类
  ↓
P0-3  闸门验收（SegMAN-ROI / weak 类 F1 / mIoU 不崩）
  ↓（未达标）
P0-4  可选加深（加长 iter / copy-paste / 结构 loss）
```

---

#### 步骤 P0-0 — 基线审计（必做，约 0.5 天）

**目的**：确认「该优先改哪些类、改分割是否有效」；避免 finetune 后 mIoU 微涨但 SegMAN-ROI 不变。

**P0-0-1 — 导出 v2@6k per-class IoU**

**目的**：得到与分类 weak 类一一对应的分割基线。

```bash
cd D:\SegMAN-main\SegMAN\segmentation
# Docker: cd /workspace/segman/segmentation

python tools/test.py \
  local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py \
  --checkpoint outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --eval mIoU \
  --work-dir outputs/trans10k_lass_mmscope_balanced_v2/eval_p0_baseline
```

**验收**：生成 `eval_p0_baseline/eval_single_scale_*.json`；记录 `IoU.shelf / IoU.door / IoU.box / IoU.freezer / IoU.bowl / mIoU`。

**P0-0-2 — 与 baseline 对比表（可选脚本）**

**目的**：快速看各类相对 Trans10K 官方 baseline 的升降。

```bash
cd D:\SegMAN-main\SegMAN\segmentation

python scripts/compare_miou_vs_baseline.py \
  outputs/trans10k_lass_mmscope_balanced_v2/eval_p0_baseline/eval_single_scale_*.json
```

**P0-0-3 — ROI 级「分割→分类」落差表（已有 T2 数据，可复算）**

**目的**：锁定 **SegMAN-ROI 瓶颈类**（见本文档 §3.5 C-4 表）。

```bash
cd D:\SegMAN-main\SegMAN
# 已有评测目录：
#   t2_unfreeze2_noweight/eval_gt_roi/summary.json
#   t2_unfreeze2_noweight/eval_segman_roi/summary.json
# 对比 per_class.f1，|Seg−GT|>15pt 的类进入 P0-1 加权名单
```

**P0-0-4 — 目视 SegMAN 弱类 mask（建议 10 张/类）**

**目的**：区分 **漏分割 / 边界错 / door 与 wall 粘连**。

```bash
cd D:\SegMAN-main\SegMAN\segmentation

# 彩色叠加（仅目视，不可用于 build_roi）
python tools/test.py \
  local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py \
  --checkpoint outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --eval mIoU \
  --show-dir outputs/trans10k_lass_mmscope_balanced_v2/vis_p0_weak \
  --show-class-names
```

人工抽查含 **shelf / door / box / freezer** 的 val 帧，记录主要错误模式（写入 `outputs/p0_weak_audit.md` 即可）。

**P0-0 产出**：

| 文件 | 说明 |
|------|------|
| `eval_p0_baseline/eval_single_scale_*.json` | 分割 per-class IoU 基线 |
| `p0_weak_audit.md`（自建） | 弱类 mask 目视结论 |
| weak 类名单 | 默认：**shelf, freezer, box, bowl, door** |

---

#### 步骤 P0-1 — 分割弱类 finetune（约 1～3 天）

**目的**：在 **不推翻 v2@6k 主权重** 的前提下，从 `iter_6000.pth` **短程微调**，抬高 weak 类 IoU 与实例 mask 质量；**mIoU 允许 ±0.5 pt 波动，但 weak 类 IoU 须升**。

**P0-1-1 —  fork 配置（勿改原 v2 文件）**

**目的**：只提高 weak 类 loss 权重，控制 door/wall 竞争。

在 `segmentation/local_configs/segman_trans/` 新建 **`segman_b_trans10k_lass_balanced_v2_p0weak.py`**（示例权重，可在 P0-0 后微调）：

```python
# P0 weak-class finetune: from balanced_v2 @ iter_6000
_base_ = ['./segman_b_trans10k_lass_balanced_v2.py']

_TRANS10K_CLASS_WEIGHT = [
    1.0,   # background
    1.15,  # box      ↑
    1.10,  # bottle
    1.08,  # window   ↑
    1.12,  # eyeglass
    1.15,  # freezer  ↑
    1.10,  # jar_kettle
    1.15,  # door     ↑
    1.0,   # cup
    0.95,  # wall     ↓ 略降，缓解 door/wall 互抢
    1.12,  # bowl     ↑
    1.20,  # shelf    ↑ 最高
]

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=_TRANS10K_CLASS_WEIGHT),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                naive_dice=True,
                loss_weight=0.18,  # 略增 Dice，利于小目标
                loss_name='loss_dice'),
            dict(
                type='BowlAntiCupLoss',
                bowl_class_index=10,
                cup_class_index=8,
                loss_weight=0.25,
                loss_name='loss_bowl_ac',
                ignore_index=255),
        ],
    ),
)

# 短程微调：小 lr，少 iter
optimizer = dict(_delete_=True, type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01)
runner = dict(type='IterBasedRunner', max_iters=4000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
data = dict(samples_per_gpu=2, workers_per_gpu=2)
```

**P0-1-2 — 启动 finetune（从 iter_6000 续训）**

**目的**：保留 v2 已学表征，仅调整 weak 类决策边界。

```bash
cd D:\SegMAN-main\SegMAN\segmentation
# Docker:
#   source activate segman
#   cd /workspace/segman/segmentation

python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2_p0weak.py \
  --work-dir outputs/trans10k_lass_mmscope_balanced_v2_p0weak \
  --load-from outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth
```

| 参数 | 值 | 目的 |
|------|-----|------|
| `--load-from iter_6000.pth` | v2 权重 | 热启动，避免从头 80k |
| `max_iters 4000` | 短程 | 降低过拟合、控制工期 |
| `lr 1e-5` | 小于 v2 初训 | 保护已有 mIoU |
| `class_weight` | shelf/door/box↑ wall↓ | 对准 C-4 弱类 |

**监控**：`tail -f outputs/trans10k_lass_mmscope_balanced_v2_p0weak/*.log`；每 2000 iter 自动 eval。

**P0-1-3 — 选取 checkpoint（勿盲目取 last）**

**目的**：mIoU 最高 ≠ SegMAN-ROI 最高；须 **weak 类 IoU 综合最好**。

```bash
cd D:\SegMAN-main\SegMAN\segmentation

# 对各 iter checkpoint 跑 eval
for ckpt in iter_2000 iter_4000; do
  python tools/test.py \
    local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2_p0weak.py \
    --checkpoint outputs/trans10k_lass_mmscope_balanced_v2_p0weak/${ckpt}.pth \
    --eval mIoU \
    --work-dir outputs/trans10k_lass_mmscope_balanced_v2_p0weak/eval_${ckpt}
done

python scripts/compare_miou_vs_baseline.py \
  outputs/trans10k_lass_mmscope_balanced_v2_p0weak/eval_iter_2000/eval_single_scale_*.json
python scripts/compare_miou_vs_baseline.py \
  outputs/trans10k_lass_mmscope_balanced_v2_p0weak/eval_iter_4000/eval_single_scale_*.json
```

**选用规则**：

1. **mIoU ≥ 81.0%**（相对 v2@6k 81.80% **下降不超过 ~0.8 pt**）；
2. **IoU.shelf + IoU.door + IoU.box** 三者平均 **≥ P0-0 基线 +1.0 pt**；
3. **IoU.wall 下降 ≤ 1.0 pt**（防止 wall 崩溃引发新混淆）。

记选定权重为 **`iter_XXXX_p0weak.pth`**（下文统称 `P0_SEG_CKPT`）。

---

#### 步骤 P0-2 — 重导 SegMAN-ROI 数据（约 0.5 天）

**目的**：用 **新分割 mask** 生成部署向 ROI；**分类权重不变**，隔离「分割增益」。

**P0-2-1 — 导出 class-id 预测 mask**

**目的**：供 `build_roi_dataset.py --mask-source segman` 使用（**禁止** `--show-dir` RGB 图）。

```bash
cd D:\SegMAN-main\SegMAN

python transgrasp/data/export_sem_seg_preds.py \
  --config segmentation/local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2_p0weak.py \
  --checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2_p0weak/P0_SEG_CKPT.pth \
  --data-root segmentation/data/trans10k \
  --split val \
  --out-dir segmentation/outputs/trans10k_lass_mmscope_balanced_v2_p0weak/pred_sem_seg_val
```

将 `P0_SEG_CKPT.pth` 替换为 P0-1-3 实际文件名（如 `iter_4000.pth`）。

**验收**：约 **1000** 张 `{stem}.png`，单通道 class-id（目视偏黑属正常）。

**P0-2-2 — 裁 SegMAN-ROI（新目录，勿覆盖旧数据）**

**目的**：与 v2@6k ROI **并列对比**。

```bash
cd D:\SegMAN-main\SegMAN

python transgrasp/data/build_roi_dataset.py \
  --data-root segmentation/data/trans10k \
  --split val \
  --mask-source segman \
  --pred-dir segmentation/outputs/trans10k_lass_mmscope_balanced_v2_p0weak/pred_sem_seg_val \
  --out-root data/trans10k_roi_segman_p0weak/val \
  --bbox-pad 0.15 \
  --min-area 64
```

**验收**：记录 ROI 数（v2@6k 为 **3233**）；对比 weak 类实例数是否增加（尤其 shelf/door recall 代理指标）。

**P0-2-3 — 统计 ROI 分布（可选）**

**目的**：确认 finetune 后 weak 类 **检出 ROI 数** 是否上升。

```bash
cd D:\SegMAN-main\SegMAN

python transgrasp/data/stats_roi_dataset.py \
  --root data/trans10k_roi_segman_p0weak \
  --split val
```

与 `data/trans10k_roi_segman/val` 对比各类 `count`。

---

#### 步骤 P0-3 — 重评分类 + 全链路闸门（约 0.5 天）

**目的**：**固定** `deliver_classifier_best.pth`，只换 ROI，测 SegMAN-ROI 纯增益。

**P0-3-1 — SegMAN-ROI 评测（P0 主指标）**

```bash
cd D:\SegMAN-main\SegMAN
# Docker: cd /workspace/segman && conda activate segman

python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint outputs/openclip_classifier/deliver_classifier_best.pth \
  --roi-root data/trans10k_roi_segman_p0weak \
  --split val \
  --report-dir outputs/openclip_classifier/p0weak_eval_segman_roi
```

**P0-3-2 — 与 v2@6k 基线对比**

| 指标 | v2@6k 基线 | P0 目标 | 回退条件 |
|------|------------|---------|----------|
| SegMAN-ROI Acc | **64.61%** | **≥68%**（stretch ≥70%） | <64.61% 回退 |
| macro F1 | 58.37% | ≥62% | 下降则回退 |
| shelf F1 | 24.7% | **≥30%**（+5 pt） | — |
| door F1 | 46.6% | **≥52%**（+5 pt） | — |
| GT-ROI Acc | 74.88% | 不变（同 checkpoint） | — |
| **ΔAcc** | −10.27 pt | **缩小**（如 ≤−8 pt） | — |
| 分割 mIoU | 81.80% | **≥81.0%** | <80.5% 回退 |

**P0-3-3 — 混淆矩阵对比**

**目的**：验证 wall→door / shelf→wall 是否减少。

```bash
# 对比以下两份 confusion_matrix.json：
#   t2_unfreeze2_noweight/eval_segman_roi/confusion_matrix.json  （基线）
#   p0weak_eval_segman_roi/confusion_matrix.json                 （P0）
```

**P0-3 通过 → 更新交付**：

```bash
cd D:\SegMAN-main\SegMAN

# 仅当 P0-3 闸门全部通过时：
cp segmentation/outputs/trans10k_lass_mmscope_balanced_v2_p0weak/P0_SEG_CKPT.pth \
   segmentation/outputs/deliver_segman_p0weak.pth

cp -r data/trans10k_roi_segman_p0weak data/trans10k_roi_segman_deliver

# 更新 deliver_manifest.json 中 segmentation 与 segman_roi 指标
```

**未通过**：保留 v2@6k + `deliver_classifier_best.pth`，进入 P0-4 或调整 P0-1 权重。

---

#### 步骤 P0-4 — 可选加深（P0-3 未达标时，约 2～4 天）

**目的**：在 P0-1 无效或 weak 类 IoU 升但 ROI Acc 不变时尝试。

| 子步骤 | 做法 | 目的 |
|--------|------|------|
| **P0-4a** | `max_iters` 4000→8000，lr 仍 1e-5 | 弱类 IoU 继续升 |
| **P0-4b** | 仅对 shelf/box/door 做 **Copy-Paste** 增广（Trans10K train） | 小样本类 |
| **P0-4c** | 新增 **door–wall 边界 loss**（自定义 decode loss 或在 mmscope 提 boundary_weight） | 减 door↔wall 粘连 |
| **P0-4d** | 对 **误分割帧** hard mining 重训（P0-0 目视列表） | 精准补弱 |

每做一项 **重复 P0-2 + P0-3**，一次只改一个变量。

---

#### P0 验收清单（汇总）

| # | 条件 | 通过标准 |
|---|------|----------|
| 1 | 分割 mIoU | ≥ **81.0%**（相对 81.80% 最多约 −0.8 pt） |
| 2 | weak 类 IoU | shelf/door/box **平均 +1 pt** vs P0-0 |
| 3 | SegMAN-ROI Acc | **≥68%**（最低 **≥65%** 且比 64.61% 明确提升） |
| 4 | shelf 或 door F1 | 至少一类 **+5 pt**（SegMAN-ROI） |
| 5 | GT-ROI | 仍 ~74.88%（分类 ckpt 未变） |
| 6 | 产物归档 | 新 seg ckpt + `trans10k_roi_segman_p0weak` + eval JSON |

**预期**：SegMAN-ROI **+3～6 pt**（→ **68%～71%**）；**GT-ROI 不变**；全链路 80% 仍难，但 **ΔAcc 显著缩小**。

**工期**：P0-0～P0-3 约 **2～4 天**；含 P0-4 约 **5～7 天**。

**风险与回退**：

| 风险 | 现象 | 处理 |
|------|------|------|
| mIoU 升、ROI Acc 不变 | mask 像素好但实例错 | 查 door/wall 粘连；加强 P0-4c |
| wall IoU 崩 | door 升 wall 大降 | wall 权重回调至 1.0 |
| 整体 mIoU 降 >1 pt | 过拟合 weak 类 | 减 iter / 减权重幅度，回退 iter_6000 |
| ROI 数暴减 | 过严分割 | 检查 pred mask 连通域、`min-area` |

**关联脚本与文档**：

| 用途 | 路径 |
|------|------|
| 导出 pred mask | `transgrasp/data/export_sem_seg_preds.py` |
| 裁 ROI | `transgrasp/data/build_roi_dataset.py` |
| 分类评测 | `transgrasp/classification/eval_openclip_classifier.py` |
| IoU 对比 | `segmentation/scripts/compare_miou_vs_baseline.py` |
| §3-3 流程 | 《OpenCLIP_细分类训练与优化指南.md》步骤 3-3 |

---

### 5.3 P1 — T2 加深 + 持久化 encoder 权重

**总目的**：T2 仅解冻 ViT **最后 2 个 block**，且 `best.pth` **未保存 encoder 微调权重**；P1 在 T2 分类头基础上 **加深至 4 block**，并将 **encoder 可训练部分写入 checkpoint**，使推理/续训不再依赖「重新跑 T2 微调」。

**前置（固定）**：

| 组件 | 路径 | 说明 |
|------|------|------|
| 分类起点 | `outputs/openclip_classifier/deliver_classifier_best.pth` | T2 **74.88% / 64.61%** |
| ROI 数据 | `data/trans10k_roi_gt` | bbox-pad **0.15**，与 §9 一致 |
| 环境 | Docker `segman_train`，conda `segman` | 与分割/P0 相同 |

**P1 总流程**（两阶段，2026-05-26 实测必需）：

```text
P1-0  代码：checkpoint_utils 支持 encoder 存取
  ↓
P1-1  warmup：unfreeze 2 block，重建 T2 visual tail + 存 encoder
  ↓
P1-2  deepen：unfreeze 4 block，低 lr 短程微调
  ↓
P1-3  GT-ROI + SegMAN-ROI 评测 + 闸门
```

> **为何两阶段？** 直接从 `deliver_classifier_best.pth` 解冻 4 block 时，encoder 仍为 LAION 预训练（T2 未存权重），首轮 val Acc 仅 **~73.8%** 且 3 epoch 早停。Warmup 写入 encoder 后，stage-2 **epoch 0 即达 74.91%**。

---

#### 步骤 P1-0 — encoder 持久化（代码）

**改动文件**：

| 文件 | 改动 |
|------|------|
| `transgrasp/classification/checkpoint_utils.py` | `encoder_trainable_state_dict()` / `load_encoder_trainable_state()`；`save_checkpoint(..., encoder=)` |
| `transgrasp/classification/train_openclip_classifier.py` | 保存/加载 encoder；未破基线时将 `last.pth` 复制为 `best.pth` 供诊断 |
| `transgrasp/classification/eval_openclip_classifier.py` | 评测时加载 `ckpt['encoder']` |

---

#### 步骤 P1-1 — warmup（unfreeze 2 block）

**配置**：`transgrasp/classification/configs/p1_warmup_unfreeze2.yaml`

| 超参 | 值 | 说明 |
|------|-----|------|
| `unfreeze_last_blocks` | **2** | 与 T2 一致 |
| `lr` / `head_lr` | **5e-6** / **1e-4** | 与 T2 一致 |
| `epochs` / `patience` | 8 / 4 | |
| `weight_decay` | 0.05 | |
| `resume` | `deliver_classifier_best.pth` | 仅加载 head；encoder 从 LAION 初始化 |

**命令**（Docker 内 `/workspace/segman`）：

```bash
python transgrasp/classification/train_openclip_classifier.py \
  --config transgrasp/classification/configs/p1_warmup_unfreeze2.yaml \
  --no-class-weights
```

**warmup 训练曲线**（`p1_warmup_unfreeze2/history.json`）：

| epoch | train_loss | val_acc | macro_f1 |
|-------|------------|---------|----------|
| 0 | 1.0389 | 73.24% | 71.30% |
| 1 | 0.9763 | 72.62% | 71.73% |
| 2 | 0.9220 | 73.46% | 72.16% |
| 3 | 0.8639 | **73.69%** | 71.70% |

- epoch 3 触发 patience=4 早停；**未超过 T2 基线 74.88%**。
- 已将 `last.pth` → `best.pth`，**含 24 个 encoder tensor**，供 stage-2 加载。

**work-dir**：`outputs/openclip_classifier/p1_warmup_unfreeze2/`

---

#### 步骤 P1-2 — deepen（unfreeze 4 block）

**配置**：`transgrasp/classification/configs/p1_unfreeze4_noweight.yaml`

| 超参 | 值 | 说明 |
|------|-----|------|
| `unfreeze_last_blocks` | **4** | 比 T2 多 2 个 block |
| `lr` / `head_lr` | **2e-6** / **5e-5** | 低于 T2，防过拟合 |
| `epochs` / `patience` | 10 / 3 | |
| `resume` | `p1_warmup_unfreeze2/best.pth` | head + encoder |

**deepen 训练曲线**（`p1_unfreeze4_noweight/history.json`）：

| epoch | train_loss | val_acc | macro_f1 | 备注 |
|-------|------------|---------|----------|------|
| **0** | 0.7980 | **74.91%** | 72.09% | 最佳 Acc |
| **1** | 0.7362 | **74.91%** | **72.54%** | **最佳 macro-F1 → 保存 best.pth** |
| 2 | 0.6853 | 74.36% | 71.46% | |
| 3 | 0.6429 | 74.56% | 71.88% | |
| 4 | 0.6130 | 74.59% | 71.56% | patience 早停 |

**选定 checkpoint**：`outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth`（epoch **1**，含 encoder）。

---

#### 步骤 P1-3 — 评测与闸门

**一键脚本**：

```bash
docker exec segman_train bash -lc \
  'source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman && \
   cd /workspace/segman && bash scripts/run_p1_train.sh'
```

**整体指标（P1 vs T2）**：

| 指标 | T2 基线 | P1 | Δ | 闸门 |
|------|---------|-----|---|------|
| GT-ROI Acc | **74.88%** | **74.91%** | +0.03 pt | ≥74.88% ✅ |
| GT macro-F1 | 72.19% | **72.54%** | +0.35 pt | 不降 ✅ |
| SegMAN-ROI Acc | 64.61% | **65.73%** | +1.12 pt | — |
| SegMAN macro-F1 | 58.37% | **60.31%** | +1.94 pt | — |
| ΔAcc (GT−Seg) | −10.27 pt | **−9.18 pt** | 缩小 1.09 pt | — |

**GT-ROI per-class F1（T2 → P1）**：

| 类 | T2 F1 | P1 F1 | Δ | 备注 |
|----|-------|-------|---|------|
| door | 63.51% | **64.77%** | +1.26 pt | 结构类略升 |
| wall | 77.46% | 77.25% | −0.21 pt | 基本持平 |
| window | 51.21% | **52.63%** | +1.42 pt | 仍 weak |
| shelf | 50.98% | 49.12% | −1.86 pt | 仍 weak |
| cup | 91.72% | 91.60% | −0.12 pt | 强类稳定 |
| eyeglass | 92.74% | 92.22% | −0.52 pt | 强类稳定 |

**SegMAN-ROI per-class F1（T2 → P1）**：

| 类 | T2 F1 | P1 F1 | Δ |
|----|-------|-------|---|
| door | 46.58% | **51.88%** | **+5.30 pt** |
| wall | 69.33% | 69.89% | +0.56 pt |
| shelf | 24.69% | **33.52%** | +8.83 pt |
| window | 46.39% | 37.07% | −9.32 pt |

**P1 结论**：

1. **闸门通过**：GT-ROI Acc / macro-F1 不低于 T2；SegMAN-ROI **意外 +1.12 pt**（encoder 加深对部署侧有小幅增益）。
2. **未达 80%**：Acc 仅 +0.03 pt，印证 §3.3「T2 后边际收益递减」；**5 pt 缺口无法靠加深 block 单独补上**。
3. **交付状态**：正式 deliver 仍为 `deliver_classifier_best.pth`；P1 `best.pth` 已含 encoder，**可作为 P2 起点与候选升级**。
4. **产物目录**：
   - `outputs/openclip_classifier/p1_warmup_unfreeze2/`
   - `outputs/openclip_classifier/p1_unfreeze4_noweight/`（含 `eval_gt_roi/`、`eval_segman_roi/`）

**风险复盘（首轮单阶段失败）**：

| 现象 | 原因 | 处理 |
|------|------|------|
| 直接 unfreeze 4 → Acc 61.81% 量级 | encoder 未 warm-up，head 与 visual 不匹配 | 改为两阶段 |
| warmup 未超 74.88% | T2 encoder 权重已丢失，需重新适配 | 仍写出 encoder 供 stage-2 |
| stage-2 epoch 1 后 val 下降 | 4 block 过拟合 | patience=3 早停，取 epoch 1 |

**工期**：实测 **~34 min**（Docker，含 warmup + deepen + 双 ROI eval）。

---

### 5.4 P2 — 层次分类（针对 door / wall / window）

**总目的**：T2/P1 混淆矩阵显示，GT val 上 **door↔wall 双向错分约 443 次**（占 door 样本 67%、wall 17%），单头 11 类 forced choice 不合理。P2 将 **结构类 {door, wall, window}** 与 **物体类 {其余 8 类}** 解耦，用 **级联二阶段** 降低结构混淆，目标 GT-ROI **+2～4 pt**（→ **77%～79%**）。

**前置**：

| 组件 | 路径 | 说明 |
|------|------|------|
| P1 checkpoint | `outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth` | GT **74.91%**，含 encoder |
| ROI 数据 | `data/trans10k_roi_gt` / `data/trans10k_roi_segman` | 与 P1 相同 |
| 混淆证据 | `t2_unfreeze2_noweight/eval_gt_roi/confusion_matrix.json` | door→wall **226**，wall→door **217** |

**结构类错分基线（T2 GT val，3105 ROI）**：

| 错分对 | 次数 | 占该类 support |
|--------|------|----------------|
| door → wall | **226** | 34.1%（226/663） |
| wall → door | **217** | 16.8%（217/1290） |
| window → wall | **67** | 51.5%（67/130） |
| shelf → wall | 21 | — |
| 结构类样本合计 | **2083** | 占 val **67.1%** |

**P2 总流程**：

```text
P2-0  基线审计 + 子集标签定义
  ↓
P2-1  数据：structure/object 二分类集 + 3 类 structure 专集
  ↓
P2-2  训练 Stage-1 路由头（structure vs object）
  ↓
P2-3  训练 Stage-2 结构专头（door / wall / window）+ 保留 P1 物体 logits
  ↓
P2-4  级联推理脚本 + GT/SegMAN 双 ROI 评测
  ↓
P2-5  闸门验收 vs P1/T2
```

---

#### 步骤 P2-0 — 基线审计与标签方案

**P2-0-1 — 确认结构类分组**

```python
STRUCTURE = {'door', 'wall', 'window'}   # 3 类，val support 2083
OBJECT    = {'box','bottle','cup','jar_kettle','bowl',
             'freezer','shelf','eyeglass'}  # 8 类，val support 1022
```

**P2-0-2 — 复算 P1 结构类 F1 缺口**（相对 80% 总 Acc 的贡献估算）

| 类 | P1 GT F1 | 目标 | 若 +10 pt 对总 Acc 贡献（粗估） |
|----|----------|------|--------------------------------|
| door | 64.77% | ≥75% | ~+1.5 pt |
| wall | 77.25% | ≥80% | ~+0.5 pt |
| window | 52.63% | ≥65% | ~+0.8 pt |

> 三者 F1 各 +10 pt 合计约 **+2～3 pt** 总 Acc，与 §5.4 预期一致；**需 P2+P3 叠加才有望触达 80%**。

**P2-0-3 — 选定级联策略**

| 方案 | 推理 | 优点 | 缺点 |
|------|------|------|------|
| **A 硬级联（推荐）** | Stage1 → 若 structure 则 Stage2-S，否则 P1 8 类头 | 实现简单、可解释 | Stage1 错则 cascade 错 |
| B 软融合 | `P(11) ∝ P(S|x)·P(c|S,x)` 对结构类相乘 | 可回退 | 需校准温度 |
| C 文本 prompt 辅助 | CLIP 文本 embedding 加 logit bias | 零样本先验 | 需调 prompt |

**P2 默认采用方案 A**；B/C 作 P2-4 可选 ablation。

---

#### 步骤 P2-1 — 子集数据准备

**目的**：从现有 `labels.csv` 派生二阶段标签，**不复制图像**（仅新 CSV + meta）。

**P2-1-1 — 脚本** `transgrasp/data/build_hierarchical_roi_labels.py` ✅ **已实现（2026-05-26）**

**一键执行**：

```bash
docker exec segman_train bash -lc \
  'source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman && \
   cd /workspace/segman && bash scripts/run_p2_1_build_hier_roi.sh'
```

**P2-1 实测结果**：

| 数据集 | split | total | structure | object |
|--------|-------|-------|-----------|--------|
| `trans10k_roi_gt_hier` | train | **15746** | 10407 (66.1%) | 5339 (33.9%) |
| `trans10k_roi_gt_hier` | val | **3105** | **2083 (67.1%)** | **1022 (32.9%)** ✅ |
| `trans10k_roi_segman_hier` | val | **3233** | 2034 (62.9%) | 1199 (37.1%) |

> val structure **2083** / object **1022** 与 §5.4 验收一致；`trans10k_roi_segman` 无 train split，仅生成 val。

- 图像：**symlink** 至源 ROI `images/`（不复制）
- meta：`meta/manifest.json`、`stage1_groups.txt`、`stage2_structure.txt`、`object_classes.txt`

**输出目录结构**：

```text
data/trans10k_roi_gt_hier/
  meta/
    classes.txt              # 原 11 类（保留索引对齐）
    stage1_groups.txt        # structure | object
    stage2_structure.txt     # door | wall | window
  train/
    labels.csv               # 原字段 + stage1_label, stage2_label
  val/
    labels.csv
```

**`labels.csv` 新增列**：

| 列 | 取值 | 说明 |
|----|------|------|
| `stage1_label` | `structure` / `object` | 二分类路由 |
| `stage2_label` | `door`/`wall`/`window` 或空 | 仅 structure 样本有值 |
| `object_label` | 8 类名或空 | 仅 object 样本有值 |

**命令（规划）**：

```bash
cd D:\SegMAN-main\SegMAN
# Docker: cd /workspace/segman

python transgrasp/data/build_hierarchical_roi_labels.py \
  --roi-root data/trans10k_roi_gt \
  --out-root data/trans10k_roi_gt_hier

python transgrasp/data/stats_roi_dataset.py \
  --root data/trans10k_roi_gt_hier --split val
```

**验收**：val 上 `structure` 样本数 ≈ **2083**，`object` ≈ **1022**；与 P1 GT val 3105 一致。

**SegMAN-ROI 同步**（P2-5 需要）：

```bash
python transgrasp/data/build_hierarchical_roi_labels.py \
  --roi-root data/trans10k_roi_segman \
  --out-root data/trans10k_roi_segman_hier
```

---

#### 步骤 P2-2 — Stage-1：structure vs object 路由头

**目的**：训练轻量二分类头，将结构类与物体类 **在决策层分开**。

**P2-2-1 — 训练入口** `transgrasp/classification/train_hier_stage1.py` ✅ **已实现（2026-05-26）**

**一键执行**：

```bash
bash scripts/run_p2_2_train_stage1.sh
```

**P2-2 首轮实测**（`p2_stage1_router/`，冻结 P1 encoder）：

| epoch | val Acc | structure recall | object recall |
|-------|---------|------------------|---------------|
| best (6) | **95.91%** | **98.61%** | **90.41%** |

| 闸门项 | 目标 | 实测 | 判定 |
|--------|------|------|------|
| val Acc | ≥95% | 95.91% | ✅ |
| structure recall | ≥98% | 98.61% | ✅ |
| object recall | ≥92% | 90.41% | ⚠️ 未达标 |

- checkpoint：`outputs/openclip_classifier/p2_stage1_router/best.pth`
- 原因：结构类占 **67%**，模型略偏向预测 structure，部分 object 被误路由。
- **补救**：`--balance-stage1` 重训 → `p2_stage1_router_balanced/`（见 `run_p2_3_train_stage2.sh`）

**P2-2 balanced 重试**（`p2_stage1_router_balanced/`，object CE 权重 **1.949×**）：

| 指标 | 首轮 router | balanced @ best (ep2) | balanced @ last (ep6) |
|------|-------------|----------------------|------------------------|
| val Acc | 95.91% | 95.30% | 94.81% |
| structure recall | 98.61% | 97.65% | 96.06% |
| object recall | 90.41% | 90.51% | **92.27%** |

- balanced 末 epoch object recall 达 **92.27%**，但 best checkpoint 仍 **gate_pass=false**。
- **级联评测采用**：`p2_stage1_router/best.pth`（Acc 更高、structure recall 更好）。

**P2-2-1 — 原规划命令**：

| 超参 | 值 | 说明 |
|------|-----|------|
| 起点 | `p1_unfreeze4_noweight/best.pth` | 共享 encoder |
| `unfreeze_last_blocks` | **0**（冻结 encoder）或 **2** | 先冻结，不稳再解冻 |
| head | linear，**2 类** | structure / object |
| `lr` / `head_lr` | **1e-4** / **1e-4** | 仅训路由头 |
| `epochs` / `patience` | 15 / 4 | |
| `class_weights` | **否** | 结构类 67% 略不平衡，可接受 |

**命令（规划）**：

```bash
python transgrasp/classification/train_hier_stage1.py \
  --roi-root data/trans10k_roi_gt_hier \
  --resume outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth \
  --work-dir outputs/openclip_classifier/p2_stage1_router \
  --freeze-encoder \
  --epochs 15 --patience 4 --batch-size 64
```

**P2-2-2 — Stage-1 验收**

| 指标 | 目标 | 回退 |
|------|------|------|
| val Acc（二分类） | **≥95%** | <92% 则检查标签或解冻 2 block |
| structure recall | **≥98%** | 过低会导致 door/wall/window 漏进 object 头 |
| object recall | **≥92%** | 过低会导致物体被误送结构专头 |

**work-dir**：`outputs/openclip_classifier/p2_stage1_router/`

---

#### 步骤 P2-3 — Stage-2：结构三分类专头 + 物体头

**目的**：在 structure 子集上专训 **door/wall/window**；物体类 **复用 P1 11 类头中 8 类 logits**（或单独 8 类头）。

**P2-3-1 — 结构专头** `train_hier_stage2_structure.py` ✅ **已实现（2026-05-26）**

**配置**：`transgrasp/classification/configs/p2_stage2_structure.yaml`  
**数据**：`HierStage2StructureDataset`（仅 `stage1_label=structure`，train **10407** / val **2083** ROI）

**一键**：

```bash
bash scripts/run_p2_3_train_stage2.sh
```

| 超参 | 值 | 说明 |
|------|-----|------|
| 训练集 | `stage1_label=structure` 的 train 子集 | 约 12k ROI（估） |
| 类别 | door / wall / window | 3 类 |
| 起点 | 同 P1 encoder（可共享 stage1 的 encoder 权重） | |
| `lr` | **5e-5** | 专头略高 |
| `epochs` / `patience` | 20 / 5 | |
| 增强（可选） | `bbox-pad 0.18` 对比实验 | 结构类需更多上下文 |

**命令（规划）**：

```bash
python transgrasp/classification/train_hier_stage2_structure.py \
  --roi-root data/trans10k_roi_gt_hier \
  --resume outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth \
  --work-dir outputs/openclip_classifier/p2_stage2_structure \
  --epochs 20 --patience 5 --batch-size 32
```

**P2-3-2 — 物体头（二选一）**

| 选项 | 做法 | 推荐 |
|------|------|------|
| **2a 复用 P1** | 推理时对 P1 logits 屏蔽 door/wall/window 三类 | ✅ 默认，零额外训练 |
| 2b 独立 8 类头 | 仅 object 子集再训一个 head | 若 2a 物体 Acc 降 >1 pt 再用 |

**P2-3-3 — Stage-2 验收（structure 专头，val 2083 子集）**

| 类 | P1 GT F1 | P2 目标 | 说明 |
|----|----------|---------|------|
| door | 64.77% | **≥72%** | 主要收益来源 |
| wall | 77.25% | **≥80%** | 少降 wall→door |
| window | 52.63% | **≥62%** | 弱类 |

| 指标 | 目标 |
|------|------|
| 结构子集 macro-F1 | **≥75%**（P1 约 72%） |
| door↔wall 互错 | 较 T2 **减半**（443 → **<220**） |

**work-dir**：`outputs/openclip_classifier/p2_stage2_structure/`

**P2-3 实测**（structure 子集 val 2083 ROI，`eval_structure_val/summary.json`）：

| 类 | P1 GT F1（全量） | P2 结构专头 F1 | Δ |
|----|------------------|----------------|---|
| door | 64.77% | **64.21%** | −0.56 pt |
| wall | 77.25% | **81.10%** | **+3.85 pt** |
| window | 52.63% | **53.68%** | +1.05 pt |
| 子集 Acc | — | **74.80%** | macro-F1 **66.33%** |

- wall 提升明显；door 未达 **≥72%** 目标；window 仍 weak。

**P2-3 训练曲线**（节选，`p2_stage2_structure/history.json`）：

| epoch | val Acc（结构子集） | macro-F1 | door F1 | wall F1 |
|-------|---------------------|----------|---------|---------|
| 0 | 67.21% | 40.92% | 38.68% | 78.23% |
| 5 | 74.46% | 64.49% | 63.26% | 81.04% |
| **9（best）** | **74.80%** | **66.33%** | **64.21%** | **81.10%** |
| 14 | 74.17% | 66.70% | 64.67% | 80.27% |

- 早停 epoch 14（patience=5）；**best @ epoch 9**。
- 工期：train **~21 min**（Docker，10407 结构 ROI × 15 epoch）。

---

#### 步骤 P2-4 — 级联推理与评测

**P2-4-1 — 评测脚本** `eval_hierarchical_classifier.py` ✅ **已实现（2026-05-26）**

**一键**：

```bash
bash scripts/run_p2_4_eval_cascade.sh
```

**硬级联逻辑**（已实现）：

```python
# 1) stage1: P(structure | x) vs P(object | x)
if stage1_pred == 'object':
    logits = p1_head(x)                    # 11 维
    logits[door_idx] = logits[wall_idx] = logits[window_idx] = -inf
    pred = argmax(logits)
else:
    pred = stage2_structure_head(x)        # 3 类 → door/wall/window
```

**P2-4-2 — GT-ROI 评测**

```bash
python transgrasp/classification/eval_hierarchical_classifier.py \
  --stage1 outputs/openclip_classifier/p2_stage1_router/best.pth \
  --stage2-structure outputs/openclip_classifier/p2_stage2_structure/best.pth \
  --object-head outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --report-dir outputs/openclip_classifier/p2_eval_gt_roi
```

**P2-4-3 — SegMAN-ROI 评测**

```bash
python transgrasp/classification/eval_hierarchical_classifier.py \
  --stage1 outputs/openclip_classifier/p2_stage1_router/best.pth \
  --stage2-structure outputs/openclip_classifier/p2_stage2_structure/best.pth \
  --object-head outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir outputs/openclip_classifier/p2_eval_segman_roi
```

**P2-4-4 — 混淆矩阵对比**

对比以下目录的 `confusion_matrix.json`：

- `p1_unfreeze4_noweight/eval_gt_roi/`（P1 基线）
- `p2_eval_gt_roi/`（P2）
- 重点：`door→wall`、`wall→door`、`window→wall` 计数

**P2-4-5 — 可选 ablation：CLIP 文本 prompt bias**

对 structure 专头增加文本先验（OpenCLIP 已有 text tower）：

```python
prompts = [
  "a door in an indoor scene",
  "a wall in an indoor scene",
  "a window in an indoor scene",
]
# logit_bias = alpha * cos_sim(image_feat, text_feat); alpha ∈ {0.1, 0.2}
```

**P2-4 实测**（`p2_stage1_router` + `p2_stage2_structure` + P1 物体头）：

| 指标 | P1 | P2 级联 | Δ | P2 目标 |
|------|-----|---------|---|---------|
| **GT-ROI Acc** | 74.91% | **75.23%** | **+0.32 pt** | ≥77% ❌ |
| GT macro-F1 | 72.54% | **71.42%** | −1.12 pt | ≥74% ❌ |
| **SegMAN-ROI Acc** | 65.73% | **67.09%** | **+1.36 pt** | ≥66% ✅ |
| SegMAN macro-F1 | 60.31% | **60.77%** | +0.46 pt | — |

**GT-ROI 结构类 F1（P1 → P2）**：

| 类 | P1 | P2 | Δ |
|----|-----|-----|---|
| door | 64.77% | 62.95% | −1.82 pt |
| wall | 77.25% | **78.09%** | +0.84 pt |
| window | 52.63% | 53.40% | +0.77 pt |

**GT-ROI 全类 F1（P2，`p2_eval_gt_roi/summary.json`）**：

| 类 | T2 | P1 | P2 | P2−P1 |
|----|-----|-----|-----|-------|
| door | 63.51% | 64.77% | 62.95% | −1.82 pt |
| wall | 77.46% | 77.25% | **78.09%** | +0.84 pt |
| window | 51.21% | 52.63% | 53.40% | +0.77 pt |
| shelf | 50.98% | 49.12% | 44.00% | −5.12 pt |
| cup | 91.72% | 91.60% | **91.35%** | −0.25 pt |
| eyeglass | 92.74% | 92.22% | 92.22% | 0 |
| bottle | 78.44% | 79.36% | 79.09% | −0.27 pt |

**SegMAN-ROI（v2@6k mask + P2 级联）**：

| 指标 | T2 | P1 | P2 | P2−P1 |
|------|-----|-----|-----|-------|
| Acc | 64.61% | 65.73% | **67.09%** | **+1.36 pt** |
| macro-F1 | 58.37% | 60.31% | **60.77%** | +0.46 pt |
| door F1 | 46.58% | 51.88% | 50.19% | −1.69 pt |
| wall F1 | 69.33% | 69.89% | **71.79%** | +1.90 pt |
| shelf F1 | 24.69% | 33.52% | 29.41% | −4.11 pt |

**door↔wall 混淆（GT val，互错次数）**：

| 模型 | door→wall | wall→door | 合计 |
|------|-----------|-----------|------|
| T2 | 226 | 217 | **443** |
| P1 | 197 | 237 | 434 |
| P2 级联 | 258 | **175** | **433** |

- wall→door **减少 42 次**（相对 P1）；door→wall **增加 61 次** → 净互错几乎不变。
- 级联 **整体 Acc 微升**，主因 wall 改善 + SegMAN 部署侧 +1.36 pt；**door 全链路仍 weak**。
- 分割侧仍用 **v2@6k**；P2 仅改分类级联，不改 mask。
- 报告：`outputs/openclip_classifier/p2_eval_gt_roi/`、`p2_eval_segman_roi/`

---

#### 步骤 P2-5 — 闸门验收

**P2-5-1 — 实测判定（2026-05-26）**

| 指标 | T2 | P1 | P2 实测 | 目标 | 判定 |
|------|-----|-----|---------|------|------|
| GT-ROI Acc | 74.88% | 74.91% | **75.23%** | ≥77% | ❌ 未达 stretch |
| GT macro-F1 | 72.19% | 72.54% | 71.42% | ≥74% | ❌ |
| door F1 (GT) | 63.51% | 64.77% | 62.95% | ≥72% | ❌ |
| wall F1 (GT) | 77.46% | 77.25% | **78.09%** | ≥78% | ✅ 临界 |
| SegMAN-ROI Acc | 64.61% | 65.73% | **67.09%** | ≥66% | ✅ |
| cup/eyeglass F1 | ~92% | ~92% | ~92% | 不降 >1 pt | ✅ |

**交付决策**：**不替换** `deliver_classifier_best.pth`；**实验最佳为 P3 单头**（GT 76.91%）；P2 级联可作 SegMAN demo 备选（67.09%）；P3+P2 级联 **75.27%** 不采用。

**P2-5-2 — 通过 → 更新交付（可选）**

```bash
# 仅当 P2-5 闸门全部通过时：
cp outputs/openclip_classifier/p2_stage2_structure/best.pth \
   outputs/openclip_classifier/deliver_hier_structure.pth
# 更新 deliver_manifest.json：method=hierarchical, 附 stage1/stage2 路径
```

**未通过**：保留 **P1** `best.pth` 或 **T2 deliver**；进入 **P3 hard mining** 或 **P2-4 prompt ablation**。

---

#### P2 待实现文件清单

| 文件 | 职责 |
|------|------|
| `transgrasp/data/build_hierarchical_roi_labels.py` | ✅ P2-1 层次标签派生 |
| `scripts/run_p2_1_build_hier_roi.sh` | ✅ P2-1 一键 GT + SegMAN |
| `transgrasp/classification/hier_dataset.py` | ✅ Stage-1 / Stage-2 Dataset |
| `transgrasp/classification/train_hier_stage1.py` | ✅ P2-2 二分类路由 |
| `transgrasp/classification/train_hier_stage2_structure.py` | ✅ P2-3 结构三分类 |
| `transgrasp/classification/eval_hierarchical_classifier.py` | ✅ P2-4 级联推理 + 11 类指标 |
| `scripts/run_p2_4_eval_cascade.sh` | ✅ P2-4 一键 GT + SegMAN |
| `transgrasp/classification/configs/p2_stage1_router.yaml` | ✅ Stage-1 超参 |
| `transgrasp/classification/configs/p2_stage2_structure.yaml` | ✅ Stage-2 超参 |
| `scripts/run_p2_2_train_stage1.sh` | ✅ P2-2 一键 |
| `scripts/run_p2_3_train_stage2.sh` | ✅ P2-2b balanced + P2-3 |

**预期**：GT-ROI **+2～4 pt** → **77%～79%**；**80% 仍不保证**（需叠 P3）。  
**工期**：实现 **1～2 天** + 训练评测 **1～2 天**，合计 **2～4 天**。

**风险与回退**：

| 风险 | 现象 | 处理 |
|------|------|------|
| Stage1 误路由 | 物体→structure，cup 被判成 wall | 提高 stage1 object recall；软融合 |
| 结构专头过拟合 | train Acc 高、val door F1 不升 | 减 epoch；冻结 encoder |
| cascade 伤强类 | cup/eyeglass F1 降 | 用 P1 logits 掩码而非重训物体 |
| SegMAN 增益有限 | GT 升、SegMAN 不变 | 结构类错分源于 mask，并行 P0-4c |

---

#### P2 实验结果汇总（2026-05-26）

**架构**（硬级联方案 A）：

```text
ROI 图像
  → Stage-1 路由（structure / object）     [p2_stage1_router/best.pth]
       ├─ object  → P1 11 类头，屏蔽 door/wall/window  [p1_unfreeze4_noweight/best.pth]
       └─ structure → Stage-2 三分类（door/wall/window） [p2_stage2_structure/best.pth]
  → 11 类 argmax
```

**分割**：全程 **v2@6k** `iter_6000.pth`（SegMAN-ROI 来自 `data/trans10k_roi_segman`）。

**端到端指标（val）**：

| 链路 | T2 交付 | P1 | **P2 级联** | vs T2 | vs P1 | 距 80% GT |
|------|---------|-----|-------------|-------|-------|-----------|
| GT-ROI Acc | 74.88% | 74.91% | **75.23%** | +0.35 pt | +0.32 pt | **−4.77 pt** |
| GT macro-F1 | 72.19% | 72.54% | 71.42% | −0.77 pt | −1.12 pt | — |
| SegMAN-ROI Acc | 64.61% | 65.73% | **67.09%** | **+2.48 pt** | **+1.36 pt** | — |
| ΔAcc (GT−Seg) | −10.27 pt | −9.18 pt | **−8.14 pt** | 缩小 2.13 pt | 缩小 1.04 pt | — |

**P2 相对预期的差距**：

| 预期（§5.4） | 实测 | 说明 |
|--------------|------|------|
| GT +2～4 pt → 77%～79% | **+0.32 pt** → 75.23% | 未达 stretch **77%** |
| door F1 +10～15 pt（结构组） | door **−1.82 pt**（全量 GT） | 结构专头 wall 升、door 未升 |
| 结构 macro-F1 ≥75% | 子集 **66.33%**；全量 macro **71.42%** | 未达标 |

**产物与脚本索引**：

| 类型 | 路径 |
|------|------|
| 层次 ROI 数据 | `data/trans10k_roi_gt_hier/`、`data/trans10k_roi_segman_hier/` |
| Stage-1 router | `outputs/openclip_classifier/p2_stage1_router/best.pth` |
| Stage-1 balanced（备选） | `outputs/openclip_classifier/p2_stage1_router_balanced/best.pth` |
| Stage-2 structure | `outputs/openclip_classifier/p2_stage2_structure/best.pth` |
| 物体头（P1） | `outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth` |
| GT 评测 | `outputs/openclip_classifier/p2_eval_gt_roi/summary.json` |
| SegMAN 评测 | `outputs/openclip_classifier/p2_eval_segman_roi/summary.json` |
| 一键数据 | `scripts/run_p2_1_build_hier_roi.sh` |
| 一键训练 | `scripts/run_p2_2_train_stage1.sh`、`scripts/run_p2_3_train_stage2.sh` |
| 一键评测 | `scripts/run_p2_4_eval_cascade.sh` |

**交付结论**：

1. **正式 deliver**：`deliver_classifier_best.pth`（**P3**，2026-05-26 升级）+ **v2@6k** 分割。
2. **可选增强包**：P2 级联 + v2@6k → SegMAN-ROI **67.09%**（较 T2 **+2.48 pt**），可作 demo/E2E 候选。
3. **冲 80% GT-ROI**：P3 单头 **76.91%**；**P3+P2 级联 ablation 已完成** → GT **75.27%**（**−1.64 pt vs P3 单头**，级联 **不采用**）；距 80% **−3.09 pt**，下一步 **P4**。

**总工期（实测）**：P2-1～P2-4 合计约 **~4 h**（Docker：数据 <1 min；Stage-1 ~23 min；Stage-2 ~35 min；级联 eval ~3 min）。

---

### 5.5 P3 — 数据增强与 Hard Example Mining

> **状态（2026-05-26）**：✅ **P3-main 已完成，闸门 PASS**。GT-ROI **76.91%**（+2.00 pt vs P1）；SegMAN-ROI **67.49%**（+1.76 pt vs P1）。✅ **P3+P2 级联 ablation 已完成**：GT **75.27%**（**低于 P3 单头，不采用**）。详见 **§5.5**。

**总目的**：P1/P2 实测 GT 仅 **75.23%**，距 80% 仍差 **~4.8 pt**；加深模型（P1）与层次分类（P2）边际收益已很小。P3 从 **训练数据侧** 入手：对 **door↔wall / window↔wall** 等 hard pairs **过采样**，并加 **轻量增广**，在 **P1 encoder+head** 上短程续训，目标 GT-ROI **+1～2 pt**（→ **76.5%～77.5%**），为 P4 或 80% stretch 铺路。**实测达成 +2.00 pt，落入预期上沿。**

**前置（固定）**：

| 组件 | 路径 | 说明 |
|------|------|------|
| 分类起点 | `outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth` | GT **74.91%**，**含 encoder**（P3 不从 T2 热启动） |
| ROI 训练 | `data/trans10k_roi_gt/train` | **15746** ROI，`bbox-pad 0.15` |
| ROI 验证 | `data/trans10k_roi_gt/val` | **3105** ROI，**禁止**参与 hard 挖掘 |
| 混淆矩阵（挖 hard） | `p1_unfreeze4_noweight/eval_gt_roi/confusion_matrix.json` | 首选；或 `p2_eval_gt_roi/` |
| 分割（SegMAN 评测） | v2@6k `iter_6000.pth` | 不变；`data/trans10k_roi_segman/val` |

**P2 后仍待解决的错分（P2 GT val，`p2_eval_gt_roi/confusion_matrix.json`）**：

| 错分对 | 次数 | P3 策略 |
|--------|------|---------|
| door → wall | **258** | **2× 过采样** + 同类 CutMix |
| wall → door | **175** | **2× 过采样** |
| window → wall | ~50+ | **2× 过采样** |
| shelf → wall | 21 | 可选 1.5×（样本少，防过拟合） |
| door ↔ wall **合计** | **433** | 主优化目标 |

> **为何不用 class_weights？** §9 已证 T1 上 class_weights **72.62%→70.18%** 有害；P3 用 **样本级过采样 + 增广**，不用 CE 权重。

**P3 总流程**：

```text
P3-0  基线审计：P1/P2 混淆 Top 错分对 + 可选 pad 对比决策
  ↓
P3-1  从 train ROI 构建 hard-pair manifest（仅 train，按 stem 防泄漏）
  ↓
P3-2  训练增广：ColorJitter / 轻旋转 / 同类 CutMix（可选）
  ↓
P3-3  WeightedRandomSampler 续训 P1（4 block，低 lr）
  ↓
P3-4  GT-ROI + SegMAN-ROI 评测
  ↓
P3-5  闸门 vs P1 / P2；可选 ablation（pad0.18 / 无 CutMix）
```

---

#### 步骤 P3-0 — 基线审计与 hard 名单

**P3-0-1 — 导出 P1 混淆 Top 错分对**

```bash
cd D:\SegMAN-main\SegMAN
# Docker: cd /workspace/segman && conda activate segman

python transgrasp/classification/tools/summarize_confusion_pairs.py \
  --confusion outputs/openclip_classifier/p1_unfreeze4_noweight/eval_gt_roi/confusion_matrix.json \
  --topk 15 \
  --out outputs/openclip_classifier/p3_hard_audit/top_pairs_p1.json
```

**验收**：输出中 **door↔wall** 为 Top-1/2；记录 `window→wall`、`shelf→wall` 排名。

**P3-0-2 — 对照 P2 级联混淆（可选）**

```bash
python transgrasp/classification/tools/summarize_confusion_pairs.py \
  --confusion outputs/openclip_classifier/p2_eval_gt_roi/confusion_matrix.json \
  --topk 15 \
  --out outputs/openclip_classifier/p3_hard_audit/top_pairs_p2.json
```

**目的**：确认 P2 后 door→wall 是否上升（P2 实测 **258**）；P3 仍以 **纠正 door/wall 边界** 为主，不重复 P2 级联。

**P3-0-3 — 选定 P3 主配方 vs 消融**

| 子实验 | 变量 | 目的 |
|--------|------|------|
| **P3-main（默认）** | hard 2× + ColorJitter + 同类 CutMix p=0.3 | 主交付实验 |
| P3-ablate-a | 仅 hard 2×，无 CutMix | 隔离增广收益 |
| P3-ablate-b | hard 2× + **bbox-pad 0.18** 重导 ROI | 结构类需更多 context（§9C pad10 更差，方向是略增 pad） |
| P3-ablate-c | 仅 door/wall 2×，window/shelf 不加 | 减少过拟合风险 |

**规则**：**一次只改一个变量**；P3-main 通过后再做 ablation。

---

#### 步骤 P3-1 — Hard-pair manifest（仅 train）

**目的**：不复制图像；为 `WeightedRandomSampler` 生成 **每样本采样权重**。

**P3-1-1 — 脚本** `transgrasp/data/build_hard_pair_manifest.py`（✅ 已实现）

**输入**：

- `data/trans10k_roi_gt/train/labels.csv`
- `--confusion`：P1 或 P2 的 `confusion_matrix.json`（**仅用于确定错分对名称**）
- `--pairs`：默认 `door,wall;wall,door;window,wall`（分号分隔真→.pred 或双向对）
- `--boost`：hard 涉及类的 ROI 权重乘数，默认 **2.0**
- `--boost-shelf`：shelf 相关可选 **1.5**

**输出**：

```text
data/trans10k_roi_gt_p3/
  meta/
    manifest.json          # 源 ROI、boost 规则、pairs
  train/
    sample_weights.csv     # path, class_name, weight（与 labels.csv 行对齐）
  val/                     # symlink 至 ../trans10k_roi_gt/val（权重恒 1.0，不增广）
```

**`sample_weights.csv` 列**：

| 列 | 说明 |
|----|------|
| `path` | 与 `labels.csv` 一致 |
| `class_name` | 11 类名 |
| `weight` | 默认 1.0；door/wall/window hard 类 **2.0** |
| `tags` | 可选：`hard_door_wall` 等 |

**P3-1-2 — 防泄漏规则（必守）**

1. **不得**用 val 预测结果直接改 train 权重（仅允许用 **val 混淆矩阵定义「哪些类对易混」**，权重施加在 **train 中同类名** ROI 上）。
2. `src_image` stem 出现在 val 的帧，hard boost 仍可用（同一帧不同实例在 train/val 已按 Trans10K 划分）。
3. **禁止**把 val ROI 复制进 train。

**命令（规划）**：

```bash
python transgrasp/data/build_hard_pair_manifest.py \
  --roi-root data/trans10k_roi_gt \
  --out-root data/trans10k_roi_gt_p3 \
  --confusion outputs/openclip_classifier/p1_unfreeze4_noweight/eval_gt_roi/confusion_matrix.json \
  --pairs "door,wall;wall,door;window,wall" \
  --boost 2.0 \
  --boost-shelf 1.5

python transgrasp/data/stats_roi_dataset.py \
  --root data/trans10k_roi_gt_p3 --split train
```

**验收**：train 总 ROI 仍 **15746**；door/wall/window 的 **有效采样期望** ≈ 2×（加权后每 epoch 期望见 door/wall ~2× 频次）。

---

#### 步骤 P3-2 — 训练增广（可选模块）

**P3-2-1 —** `transgrasp/classification/roi_augment.py`（✅ 已实现）

| 增广 | 参数 | 说明 |
|------|------|------|
| `ColorJitter` | brightness/contrast/saturation **0.2**，hue **0.05** | 透明域反光 |
| `RandomRotation` | **±5°** | 轻微，避免大角度 |
| **同类 CutMix** | p=**0.3**，β=**1.0** | **仅同类**内 mix，防标签污染 |
| 禁用 | MixUp 跨类、RandAugment 强策略 | 11 类细粒度易破坏 |

**接入方式**：在 `train_openclip_classifier.py` 增加 `--aug p3`；增广 **仅 train**，val 仍用 OpenCLIP 标准 `preprocess_val`。

**P3-2-2 — 可选 Copy-Paste（weak 类，P3-plus）**

- 对 **shelf / window** train ROI 贴到 **door/wall 背景**（同类 mask 源图内 crop）。
- 脚本：`transgrasp/data/copy_paste_roi_augment.py`（低优先级；样本少，易过拟合）。
- **必须**检查 `src_image` 不落入 val stem 列表。

---

#### 步骤 P3-3 — 续训 P1（WeightedSampler + 低 lr）

**目的**：在 **P1 已解冻 4 block + encoder 权重** 基础上微调决策边界，避免从头 T2。

**配置**：`transgrasp/classification/configs/p3_p1_hardmining.yaml`（待建）

| 超参 | 值 | 说明 |
|------|-----|------|
| `resume` | `p1_unfreeze4_noweight/best.pth` | head + encoder |
| `roi-root` | `data/trans10k_roi_gt_p3` | 含 `sample_weights.csv` |
| `unfreeze_last_blocks` | **4** | 与 P1 一致 |
| `lr` / `head_lr` | **1e-6** / **3e-5** | 低于 P1 stage-2，防遗忘 |
| `epochs` / `patience` | **12** / **4** | |
| `batch-size` | **32** | |
| `weight-decay` | **0.05** | |
| `label-smoothing` | **0.1** | |
| `no-class-weights` | **是** | 禁止 CE class_weights |
| `aug` | **p3** | ColorJitter + CutMix |
| `weighted-sampler` | **是** | 按 `sample_weights.csv` |

**P3-3-1 — 训练入口** `train_openclip_classifier.py`（✅ 已扩展 `--aug`、`--weighted-sampler`、`--cutmix-prob`）

- `--sample-weights-csv`：train  split 下权重文件；
- `--aug {none,p3}`：增广开关；
- `WeightedRandomSampler(weights, num_samples=len(train), replacement=True)`。

**命令（规划）**：

```bash
docker exec segman_train bash -lc \
  'source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman && \
   cd /workspace/segman && bash scripts/run_p3_train.sh'
```

`scripts/run_p3_train.sh` 顺序：

1. `build_hard_pair_manifest.py`（若尚未生成）
2. `train_openclip_classifier.py --config configs/p3_p1_hardmining.yaml`
3. `eval_openclip_classifier.py` → GT + SegMAN
4. 打印 P3 闸门 JSON

**work-dir**：`outputs/openclip_classifier/p3_p1_hardmining/`

**监控**：door/wall/window 的 val F1 是否升；cup/eyeglass 是否降 >1 pt（过拟合/误伤信号）。

---

#### 步骤 P3-4 — 评测

**P3-4-1 — GT-ROI**

```bash
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint outputs/openclip_classifier/p3_p1_hardmining/best.pth \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --report-dir outputs/openclip_classifier/p3_p1_hardmining/eval_gt_roi
```

**P3-4-2 — SegMAN-ROI（v2@6k，不变）**

```bash
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint outputs/openclip_classifier/p3_p1_hardmining/best.pth \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir outputs/openclip_classifier/p3_p1_hardmining/eval_segman_roi
```

**P3-4-3 — 混淆矩阵对比**

| 对比项 | P1 基线 | P2 | **P3 实测** | 目标 |
|--------|---------|-----|-------------|------|
| 报告目录 | `p1_unfreeze4_noweight/eval_gt_roi/` | `p2_eval_gt_roi/` | `p3_p1_hardmining/eval_gt_roi/` | — |
| door→wall + wall→door | **434**（237+197） | **433**（258+175） | **396**（224+172） | **<380**（❌ 未达，仅 −38） |
| door F1 | **64.77%** | 62.95% | **66.61%** | **≥66%** ✅ |
| window→wall | 62 | 72 | **69** | 降 |

**P3-4-4 — P3 + P2 级联 ablation**（✅ 2026-05-26 已完成）

**配置**：P2 Stage-1 路由 + P2 Stage-2 结构专头 **不变**；`--object-head` 换为 `p3_p1_hardmining/best.pth`。

**命令**（输出目录独立，**不覆盖** `p2_eval_*` / `p3_p1_hardmining/eval_*`）：

```bash
bash scripts/run_p3p2_eval_cascade.sh
# 或：OBJECT=outputs/openclip_classifier/p3_p1_hardmining/best.pth bash scripts/run_p3p2_eval_cascade.sh
```

**P3+P2 实测**（`p3p2_eval_gt_roi/`、`p3p2_eval_segman_roi/`）：

| 指标 | P2 级联 | P3 单头 | **P3+P2** | Δ vs P2 | Δ vs P3 单头 |
|------|---------|---------|-----------|---------|--------------|
| GT-ROI Acc | 75.23% | **76.91%** | 75.27% | +0.04 pt | **−1.64 pt** |
| GT macro-F1 | 71.42% | **73.52%** | 71.32% | −0.10 pt | −2.20 pt |
| door F1 (GT) | 62.95% | **66.61%** | 62.95% | 0 | **−3.66 pt** |
| wall F1 (GT) | 78.09% | **79.69%** | 78.09% | 0 | −1.60 pt |
| SegMAN-ROI Acc | 67.09% | **67.49%** | 67.18% | +0.09 pt | −0.31 pt |
| door↔wall 混淆 (GT) | 433 | **396** | **433** | 0 | +37 |

**原因分析**：

1. 级联中 door/wall/window 仍走 **P2 Stage-2 专头**，door/wall F1 与 P2 **完全相同**（62.95% / 78.09%），P3 在结构类上的提升 **无法体现**。
2. object 分支虽换 P3，但 Stage-1 路由 + door/wall/window logit mask 限制下，整体 **低于 P3 单头 11 类直接预测**。
3. door↔wall 混淆 **433**（与 P2 一致），未继承 P3 单头的 **396**。

**ablation 结论**：**PARTIAL**（略超 P2 +0.04 pt，**明显低于 P3 单头**）。**实验最佳仍为 P3 单头**；级联路线除非重训 Stage-2/router，否则 **不建议采用**。

---

#### 步骤 P3-5 — 闸门验收

**P3-5-1 — 与 P1 / P2 / T2 对比**

| 指标 | T2 | P1 | P2 | P3 目标 | **P3 实测** | 判定 |
|------|-----|-----|-----|---------|-------------|------|
| GT-ROI Acc | 74.88% | 74.91% | 75.23% | **≥76.5%**（stretch **≥77%**） | **76.91%** | ✅ / ⚠️ |
| GT macro-F1 | 72.19% | 72.54% | 71.42% | **≥72.0%** | **73.52%** | ✅ |
| door F1 (GT) | 63.51% | 64.77% | 62.95% | **≥66%** | **66.61%** | ✅ |
| wall F1 (GT) | 77.46% | 77.25% | 78.09% | **≥77%** | **79.69%** | ✅ |
| SegMAN-ROI Acc | 64.61% | 65.73% | 67.09% | **≥65.5%** | **67.49%** | ✅ |
| cup/eyeglass F1 | ~92% | 91.60% / 92.22% | ~92% | **不降 >1 pt** | **92.66% / 93.48%** | ✅ |

**P3-5-2 — 阶段判定**（**适用：GT 76%～77%**）

| 结果 | 动作 | **本次** |
|------|------|----------|
| GT **≥77%** | 进入 **P4 评估** 或重复 P3 ablation 冲 78%+ | — |
| GT **76%～77%** | 保留 P3 ckpt；可选叠 **P2 级联** 或 **P3-ablate-b pad0.18** | **✅ 76.91%**（P3 单头） |
| GT **<76%** 且 ≥P1 | 记录负结果；试 **P3-ablate** 或 **P4** | — |
| GT **< P1** | **回退**；不再加大 boost / CutMix | — |

**P3-5-3 — 通过 → 更新交付（可选）**

```bash
# 仅当 GT ≥77% 且 SegMAN ≥65.5%：
cp outputs/openclip_classifier/p3_p1_hardmining/best.pth \
   outputs/openclip_classifier/deliver_classifier_p3.pth
# 更新 deliver_manifest.json（method=P3 hard mining, 附 P1 encoder 路径）
```

**未通过**：保留 **P1** 或 **T2 deliver**；文档记录 P3 负结果，进入 **P4**。

---

#### P3 文件清单（已实现）

| 文件 | 职责 | 状态 |
|------|------|------|
| `transgrasp/data/build_hard_pair_manifest.py` | train 样本权重 + p3 ROI 目录 | ✅ |
| `transgrasp/classification/roi_augment.py` | ColorJitter / 同类 CutMix | ✅ |
| `transgrasp/classification/weighted_roi_dataset.py` | 加权数据集 + PIL 增广 | ✅ |
| `transgrasp/classification/tools/summarize_confusion_pairs.py` | P3-0 混淆对统计 | ✅ |
| `transgrasp/classification/configs/p3_p1_hardmining.yaml` | P3 超参 | ✅ |
| `scripts/run_p3_train.sh` | P3-0～P3-5 一键 | ✅ |
| `scripts/run_p3p2_eval_cascade.sh` | P3+P2 级联评测（独立输出目录） | ✅ |
| `train_openclip_classifier.py` | `--aug`、`--weighted-sampler`、`--cutmix-prob` | ✅ |

**预期**：GT-ROI **+1～2 pt** → **76%～77.5%**；与 P2 叠加 **不保证** 80%，需 **P3+P4** 或 P3 效果显著。  
**实测**：GT **+2.00 pt** → **76.91%**；SegMAN **+1.76 pt** → **67.49%**；训练+评测约 **~57 min**（Docker，10 epoch + early stop）。

**风险与回退**：

| 风险 | 现象 | 处理 |
|------|------|------|
| hard 过采样过强 | train acc 高、val door 不升 | boost 2.0→1.5；去掉 CutMix |
| CutMix 标签噪声 | macro-F1 降 | 仅同类 mix；p 0.3→0.15 |
| 强类受伤 | cup/eyeglass F1 降 | 降低非 hard 类权重；缩短 epoch |
| SegMAN 不升 | GT 升、SegMAN 平 | 分割瓶颈；并行 P0-4c，分类侧仍保留 P3 |

**与 P2 / P4 关系**：

- **P2 已完**：P3 走 **单头 P1 续训** 路线。**P3+P2 级联 ablation 负结果**：GT 75.27% < P3 单头 76.91%，因 structure 仍走 P2 专头。
- **P4**：P3 后 GT **<78%** 且级联无效 → **直接启动 P4** 域适配。

#### P3 实验结果汇总（2026-05-26）

**配置**：从 P1 `best.pth`（unfreeze 4 + encoder）续训；`data/trans10k_roi_gt_p3`（door/wall/window **2×** 采样权重，shelf **1.5×**）；`--aug p3`（ColorJitter + ±5° 旋转 + 同类 CutMix p=0.3）；lr **1e-6** / head **3e-5**；epochs 12 / patience 4。

**训练曲线**（`outputs/openclip_classifier/p3_p1_hardmining/history.json`）：

| Epoch | train_loss | val_acc | val_macro_f1 |
|-------|------------|---------|--------------|
| 0 | 0.9225 | 75.14% | 72.41% |
| 1 | 0.8829 | 75.85% | 72.62% |
| 2 | 0.8565 | 75.97% | 72.67% |
| 3 | 0.8431 | 76.01% | 72.66% |
| 4 | 0.8278 | 76.17% | 73.06% |
| **5** | **0.8059** | **76.91%** | **73.52%** |
| 6 | 0.7882 | 76.81% | 73.74% |
| 7 | 0.7765 | 76.14% | 73.84% |
| 8 | 0.7603 | 76.17% | 73.26% |
| 9 | 0.7409 | 76.52% | 74.25% |

Early stop @ epoch 9（patience=4）；**best @ epoch 5**。

**P3-0 混淆审计**（`outputs/openclip_classifier/p3_hard_audit/`）：

| 来源 | door↔wall 合计 | Top 错分 |
|------|----------------|----------|
| P1 单头 | **434** | wall→door 237, door→wall 197 |
| P2 级联 | **433** | door→wall 258, wall→door 175 |
| P3 单头 | **396** | wall→door 172, door→wall 224 |
| P3+P2 级联 | **433** | 与 P2 相同（structure 仍走 Stage-2） |

**P3-1 manifest**：train ROI **15746**（不变）；door/wall/window 有效采样期望 **2×**。

**P3-4 评测**：

| 指标 | T2 | P1 | P2 | **P3** | Δ vs P1 |
|------|-----|-----|-----|--------|---------|
| GT-ROI Acc | 74.88% | 74.91% | 75.23% | **76.91%** | **+2.00 pt** |
| GT macro-F1 | 72.19% | 72.54% | 71.42% | **73.52%** | +0.98 pt |
| door F1 (GT) | 63.51% | 64.77% | 62.95% | **66.61%** | +1.84 pt |
| wall F1 (GT) | 77.46% | 77.25% | 78.09% | **79.69%** | +2.44 pt |
| window F1 (GT) | — | — | — | **53.40%** | — |
| cup F1 (GT) | ~92% | 91.60% | ~92% | **92.66%** | +1.06 pt |
| eyeglass F1 | ~93% | 93.48% | ~93% | **93.48%** | 0 pt |
| SegMAN-ROI Acc | 64.61% | 65.73% | **67.09%** | **67.49%** | +1.76 pt |
| door↔wall 混淆 (GT) | — | 434 | 433 | **396**（224+172） | −38 |

> **P3+P2 级联**（另表 §P3-4-4）：GT Acc **75.27%**，SegMAN **67.18%**，**低于 P3 单头**，不采用。

**P3-3 训练摘要**：resume P1 `best.pth`（baseline val **74.91%**）；`WeightedRandomSampler` + `--aug p3`；batch **32**；总 wall-clock **~57 min**（单 epoch ~5.5 min）。

**P3-4 逐类 F1（GT-ROI，P1 → P3）**：

| 类 | P1 F1 | P3 F1 | Δ |
|----|-------|-------|---|
| door | 64.77% | **66.61%** | +1.84 pt |
| wall | 77.25% | **79.69%** | +2.44 pt |
| window | 52.63% | **53.40%** | +0.77 pt |
| shelf | 49.12% | **51.85%** | +2.73 pt |
| cup | 91.60% | **92.66%** | +1.06 pt |
| eyeglass | 92.22% | **93.48%** | +1.26 pt |
| bottle | 79.36% | **79.46%** | +0.10 pt |
| jar_kettle | 71.88% | **71.43%** | −0.45 pt |
| box | 69.94% | **68.67%** | −1.27 pt |

**P3-4 SegMAN 逐类亮点**：wall F1 **71.86%**（P1 69.89%）；door F1 **52.60%**（P1 51.88%）；整体 Acc **67.49%** 为当前单头最高。

**P3-5 闸门**（`eval_gt_roi/gate.json`）：**PASS**

| 闸门 | 阈值 | 实测 | 结果 |
|------|------|------|------|
| GT Acc ≥ P1 | 74.91% | 76.91% | ✅ |
| GT Acc stretch | 76.5% / 77% | 76.91% | ✅ / ⚠️（未达 77%） |
| door F1 | ≥66% | 66.61% | ✅ |
| SegMAN Acc | ≥65.5% | 67.49% | ✅ |

**交付结论**：

- P3 **单头 GT 76.91%** 为当前 **最高 GT-ROI**（超 P2 级联 +1.68 pt、P1 +2.00 pt）。
- SegMAN **67.49%** 略超 P2（67.09%），为当前 **最高 SegMAN-ROI 单头**。
- **P3+P2 级联** GT **75.27%**（vs P3 单头 **−1.64 pt**），**不采用**；door/wall F1 与 P2 相同，级联无法传递 P3 结构类收益。
- GT **未达 77% stretch**，**80% 仍未达标**。
- **2026-05-26 已升级为正式 deliver**：`deliver_classifier_best.pth` ← P3 `best.pth`；T2 备份 `deliver_classifier_t2_archived.pth`。
- manifest：`outputs/openclip_classifier/deliver_p3/deliver_manifest.json`
- **下一步**：~~方案 B（P5 拒识 + 双指标结题）~~ **方案 B 已完成**（见《完整优化历程》§12）。

**产物**：

| 路径 | 说明 |
|------|------|
| `outputs/openclip_classifier/p3_p1_hardmining/best.pth` | P3 best（epoch 5，含 encoder） |
| `outputs/openclip_classifier/p3_p1_hardmining/eval_gt_roi/` | GT 评测 + gate.json |
| `outputs/openclip_classifier/p3_p1_hardmining/eval_segman_roi/` | SegMAN 评测 |
| `data/trans10k_roi_gt_p3/` | hard mining manifest |
| `scripts/run_p3_train.sh` | P3-0～P3-5 一键 |
| `outputs/openclip_classifier/p3_hard_audit/top_pairs_p1.json` | P3-0 混淆审计 |
| `outputs/openclip_classifier/p3_p1_hardmining/train.log` | P3 训练日志 |
| `outputs/openclip_classifier/p3_p1_hardmining/history.json` | P3 训练曲线 |
| `outputs/openclip_classifier/p3_p1_hardmining/eval_gt_roi/gate.json` | P3 闸门 JSON |
| `scripts/run_p3p2_eval_cascade.sh` | P3+P2 级联评测一键 |
| `outputs/openclip_classifier/p3p2_eval_gt_roi/` | P3+P2 GT 评测（**75.27%**）+ gate.json |
| `outputs/openclip_classifier/p3p2_eval_segman_roi/` | P3+P2 SegMAN 评测（**67.18%**） |

---

### 5.6 P4 — 域适配预训练（中长期）

> **状态（2026-05-26）**：✅ **P4 快速验证已完成**（WiSE-FT sweep + small contrastive）。**均未达 78% 闸门**；contrastive small 微升 **+0.13 pt** → GT **77.04%**。详见 **§5.6.1**。

**目的**：突破 LAION 通用 CLIP 在透明物体上的天花板。

**做法**（择一）：

1. **Trans10K 图像-类名对比学习**（非 LAION 规模，仅 domain adapt）：冻结 text，轻量 adapt image tower。
2. 换 **透明物体专用 backbone**（若有公开权重）替代 ViT-B-16。
3. **WiSE-FT 式** 权重插值：`θ = α·θ_P3 + (1−α)·θ_pretrained`，α∈[0.7,0.95]`，减轻遗忘。

**预期**：GT-ROI **+3～6 pt**，**有可能触达 80%**。  
**风险**：工程量大、易过拟合、需额外算力。  
**工期**：1～2 周（完整 P4）；快速验证 **~18 min**（Docker）。

#### 5.6.1 P4 快速验证（WiSE-FT + contrastive small）

**目的**：在投入完整 P4 前，验证能否从 P3 **76.91%** 快速冲到 **78%**。

**脚本**：`scripts/run_p4_validate.sh`（一键）

| 步骤 | 脚本 | 说明 |
|------|------|------|
| P4-1 | `eval_wise_ft_sweep.py` | α∈[0.5…1.0] 插值 P3 encoder 48 张量 vs LAION |
| P4-2 | `train_contrastive_adapt.py` | 8000 train ROI，4 epoch CLIP loss + 2 epoch head CE |
| P4-3 | `eval_openclip_classifier.py` | GT + SegMAN 评测 best ckpt |

**P4-1 WiSE-FT sweep**（`p4_wise_ft_sweep/sweep.json`）：

| α | GT Acc | door F1 | wall F1 |
|---|--------|---------|---------|
| 0.70 | 76.10% | 63.59% | 78.83% |
| 0.85 | 76.52% | 65.25% | 79.23% |
| **0.95** | **76.94%** | 66.46% | 79.64% |
| 1.00 (P3) | 76.91% | 66.61% | 79.69% |

**最佳 α=0.95**，较 P3 **+0.03 pt**；**未达 78%**。

**P4-2 contrastive small**（resume P3，`max_train_samples=8000`）：

| 阶段 | val Acc | 说明 |
|------|---------|------|
| contrastive ep0 | **77.04%** | best |
| contrastive ep1～3 | 76.04%→75.14% | 过拟合/遗忘 |
| head-ft 2 ep | 75.56% | 未超 ep0 |

**P4-3 最终评测**：

| 方法 | GT-ROI | SegMAN-ROI | vs P3 |
|------|--------|------------|-------|
| P3 单头（基线） | 76.91% | 67.49% | — |
| WiSE-FT α=0.95 | 76.94% | — | +0.03 pt |
| **contrastive small** | **77.04%** | **67.68%** | **+0.13 pt** |

**闸门**（`p4_validate_gate.json`）：**FAIL**（`wise_pass_78=false`，`cl_pass_78=false`）。

**结论**：

1. 轻量 WiSE-FT / 小规模 contrastive **无法补齐 ~1 pt 至 78%**，更不可能一步到 80%。
2. contrastive **epoch 0 即峰值**，继续训练反降 → 需 **early stop + 全量数据 + 更低 lr** 才值得做完整 P4。
3. **实验最高**暂记 contrastive small **77.04%**；**推荐部署仍用 P3 单头**（更稳、无 contrastive 过拟合风险）。
4. 若课题硬追 80%：需 **完整 P4**（全 train ROI、更长 schedule、或换 backbone），或接受 **P5 拒识 + 双指标交付**。

**产物**：

| 路径 | 说明 |
|------|------|
| `scripts/run_p4_validate.sh` | P4 快速验证一键 |
| `transgrasp/classification/eval_wise_ft_sweep.py` | WiSE-FT α sweep |
| `transgrasp/classification/train_contrastive_adapt.py` | 小规模 contrastive |
| `outputs/openclip_classifier/p4_wise_ft_sweep/` | sweep.json + best.pth（α=0.95） |
| `outputs/openclip_classifier/p4_contrastive_small/` | best.pth + eval |
| `outputs/openclip_classifier/p4_validate_gate.json` | 78% 闸门汇总 |

#### 5.6.2 方案 A — 完整 P4 实施步骤（冲 78%～80%）

> **前置结论（§5.6.1）**：轻量验证 **FAIL**；contrastive **epoch 0 即峰值**、继续训练 val 反降。完整 P4 的核心改动：**全量 train + 更低 lr + contrastive patience=1～2 + 可选 WiSE-FT 后处理 + 短程 head 对齐**。

**总目的**：在 P3 **76.91%** 基础上，通过 **全量 ROI–文本对比学习** 再挖 **+1～3 pt**，阶段闸门 **78%**，stretch **80%**。

**固定前置**：

| 组件 | 路径 | 说明 |
|------|------|------|
| 分类起点 | `p3_p1_hardmining/best.pth` | GT **76.91%**，含 encoder 48 张量 |
| ROI train/val | `data/trans10k_roi_gt` | train **15746** / val **3105** |
| SegMAN 评测 | `data/trans10k_roi_segman/val` | v2@6k 不变 |
| 分割 | `iter_6000.pth` | 不改 |
| Docker | `segman_train` | 与 P1～P3 相同 |

**P4-full 总流程**：

```text
P4F-0  基线锁定 + 快速验证结论复核
P4F-1  扩展 train_contrastive_adapt（patience / 全量 / yaml / 每 epoch 存盘）
P4F-2  全量 contrastive 主实验（lr sweep 2 点）
P4F-3  短程 head CE 对齐（1～2 epoch，可选）
P4F-4  WiSE-FT 后处理（对 P4F-2 best encoder 再 sweep α）
P4F-5  GT + SegMAN 双 ROI 评测 + 混淆审计
P4F-6  闸门 78% / 80%；未达则 ablation 或 P4F-7 backbone
P4F-7  （可选）ViT-B-32 / 专用 CLIP — 仅当 P4F-6 <78%
```

---

##### 步骤 P4F-0 — 基线锁定

**P4F-0-1 — 确认 P3 与快速验证产物**

```bash
# Docker 内
ls -la outputs/openclip_classifier/p3_p1_hardmining/best.pth
cat outputs/openclip_classifier/p4_validate_gate.json
cat outputs/openclip_classifier/p4_contrastive_small/contrastive_history.json
```

**验收**：P3 GT **76.91%**；small contrastive ep0 **77.04%** 可复现。

**P4F-0-2 — 记录主配方（P4-main）**

| 超参 | P4-small（已跑） | **P4-full-main（默认）** |
|------|------------------|---------------------------|
| `max_train_samples` | 8000 | **−1（全量 15746）** |
| `epochs` | 4 | **6**（配合 early stop） |
| `patience` | 无 | **1**（small 已证 ep0 最佳） |
| `lr` | 5e-7 | **2e-7**（主）；ablate **1e-7** |
| `batch_size` | 64 | **64** |
| `unfreeze_last_blocks` | 4 | **4** |
| `text_template` | transparent indoor | 同左 + ablate 多模板 |
| `head_finetune_epochs` | 2 | **1**（仅当 contrastive 升） |
| `head_lr` | 1e-5 | **3e-6** |
| `weight_decay` | 0.01 | **0.05** |

---

##### 步骤 P4F-1 — 代码扩展（✅ 已实现）

**P4F-1-1 —** `train_contrastive_adapt.py`：已扩展 `--config`、`--patience`、`--eval-every`、`--no-head-finetune`、`max_train_samples=-1`。

**P4F-1-2 —** `transgrasp/classification/configs/p4_contrastive_full.yaml` ✅

**P4F-1-3 —** `scripts/run_p4_full.sh`、`scripts/run_p4_full_eval_only.sh` ✅

---

##### P4-full 实验结果汇总（2026-05-26）

**训练**（`p4_contrastive_full/contrastive_history.json`）：

| Epoch | train_loss | val_acc | 说明 |
|-------|------------|---------|------|
| 0 | 5.0594 | **76.91%** | 与 P3 持平 |
| 1 | 4.4592 | 76.39% | patience=1 → early stop |

head-ft 1 epoch → **76.43%**（未超 ep0）；**best 仍为 ep0 = P3 水平**。

**对比**：

| 方法 | GT-ROI | SegMAN | vs P3 | vs P4-small |
|------|--------|--------|-------|-------------|
| P3 单头 | 76.91% | 67.49% | — | — |
| P4-small | **77.04%** | 67.68% | +0.13 pt | — |
| **P4-full** | 76.91% | 67.65% | 0 | **−0.13 pt** |
| WiSE-FT (P4-full) | 76.91% (α=1.0) | — | 0 | — |

**闸门**（`p4_contrastive_full/eval_gt_roi/gate.json`）：**FAIL**（78%/80% 均未达；**未超 P4-small**）。

**结论**：

1. **全量 train + 更低 lr + patience=1 未能复现 small 的 77.04%**；全量反而 **持平 P3**。
2. 推测：8000 子集 **偶然更优** 或 **全量 contrastive 过拟合更快**；继续 P4 contrastive **无边际收益**。
3. **方案 A（完整 P4 contrastive）终止**；实验最高仍为 **P4-small 77.04%**；**部署推荐 P3 单头 76.91%**（更稳）。
4. 若仍追 80%：仅 **P4F-7c 换 backbone** 或 **方案 B（P5 拒识 + 双指标结题）** 有意义。

**产物**：`outputs/openclip_classifier/p4_contrastive_full/`、`p4_full_wise_ft/`、`gate.json`

---

##### 步骤 P4F-2 — 全量 contrastive 主实验

**P4F-2-1 — 主跑**

```bash
docker exec segman_train bash -lc \
  'source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman && \
   cd /workspace/segman && bash scripts/run_p4_full.sh'
```

**`run_p4_full.sh` 核心命令（规划）**：

```bash
python transgrasp/classification/train_contrastive_adapt.py \
  --config transgrasp/classification/configs/p4_contrastive_full.yaml \
  2>&1 | tee outputs/openclip_classifier/p4_contrastive_full/train.log
```

**P4F-2-2 — lr ablation（仅当 main 未达 78%）**

| 运行 | `work_dir` | `lr` |
|------|------------|------|
| main | `p4_contrastive_full` | **2e-7** |
| ablate-low | `p4_contrastive_full_lr1e7` | **1e-7** |

**监控**（每 epoch 打印 + `history.json`）：

- val_acc 是否 **>77.04%**（超 small ep0）
- train_loss 是否持续降而 val 升（过拟合信号 → 停）
- door/wall F1 是否 **≥ P3**

**work-dir**：`outputs/openclip_classifier/p4_contrastive_full/`

**预期 wall-clock**：全量 × 6 epoch ≈ **2～4 h**（Docker 单卡）。

---

##### 步骤 P4F-3 — 短程 head CE 对齐（条件执行）

**条件**：P4F-2 best val_acc **> P3 76.91%** 且 **≤78%**。

**做法**：

- 冻结 encoder；**1 epoch** head CE，`head_lr=3e-6`，`label_smoothing=0.1`
- 若 head-ft 后 val **降** → **回退** contrastive-only best

**禁止**：head-ft **>2 epoch**（small 已证易遗忘 contrastive 增益）。

---

##### 步骤 P4F-4 — WiSE-FT 后处理

对 **P4F-2（或 P4F-3）best.pth** 再跑 α sweep（encoder 48 张量 vs LAION）：

```bash
python transgrasp/classification/eval_wise_ft_sweep.py \
  --checkpoint outputs/openclip_classifier/p4_contrastive_full/best.pth \
  --roi-root data/trans10k_roi_gt \
  --alphas "0.9,0.92,0.94,0.95,0.96,0.98,1.0" \
  --report-dir outputs/openclip_classifier/p4_full_wise_ft \
  --save-best
```

**说明**：P3 上 best α=**0.95**；P4-full 后最优 α 可能偏移，需重 sweep。

**预期**：+0～0.2 pt 额外增益。

---

##### 步骤 P4F-5 — 评测与审计

**P4F-5-1 — GT-ROI**

```bash
BEST=outputs/openclip_classifier/p4_contrastive_full/best.pth  # 或 wise-ft best
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint "${BEST}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --report-dir outputs/openclip_classifier/p4_contrastive_full/eval_gt_roi
```

**P4F-5-2 — SegMAN-ROI**

```bash
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint "${BEST}" \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir outputs/openclip_classifier/p4_contrastive_full/eval_segman_roi
```

**P4F-5-3 — 混淆对比**

```bash
python transgrasp/classification/tools/summarize_confusion_pairs.py \
  --confusion outputs/openclip_classifier/p4_contrastive_full/eval_gt_roi/confusion_matrix.json \
  --topk 15 \
  --out outputs/openclip_classifier/p4_contrastive_full/top_pairs.json
```

对比 P3：**door↔wall 合计 396** 能否再降；door F1 **66.61%** 能否保持/升。

---

##### 步骤 P4F-6 — 闸门验收

**P4F-6-1 — 对比表**

| 指标 | P3 | P4-small | **P4-full 目标** | stretch |
|------|-----|----------|------------------|---------|
| GT-ROI Acc | 76.91% | 77.04% | **≥78.0%** | **≥80.0%** |
| SegMAN-ROI Acc | 67.49% | 67.68% | **≥67.0%**（勿大幅低于 P3） | ≥68% |
| door F1 (GT) | 66.61% | — | **≥66%** | ≥68% |
| GT macro-F1 | 73.52% | 73.81% | **≥73.5%** | ≥75% |

**P4F-6-2 — 阶段判定**

| 结果 | 动作 |
|------|------|
| GT **≥80%** | 更新实验 deliver；写 manifest；可选替换正式 deliver（需评审） |
| GT **78%～80%** | 保留 P4-full ckpt；可选 P4F-7 冲 80%；或转 P5 |
| GT **77%～78%** | 记录正结果；**P4F-7a** 多模板 / **P4F-7b** 全图 contrastive |
| GT **<77.04%** | **回退 P3**；停止 P4；转 **方案 B（P5 结题）** |
| SegMAN 降 **>1 pt** | 回退；仅保留 GT 提升 run |

**P4F-6-3 — gate.json**

输出：`outputs/openclip_classifier/p4_contrastive_full/eval_gt_roi/gate.json`（对比 P3 / P4-small / T2）。

---

##### 步骤 P4F-7 — 可选进阶（仅当 P4F-6 未达 78%）

**P4F-7a — 多文本模板 ensemble（低成本）**

```python
templates = [
  "a transparent {name} in an indoor scene",
  "a photo of a {name}",
  "a {name} object behind glass",
]
# 训练时对每样本随机选一模板；或推理时 logit 平均
```

**P4F-7b — 全图 contrastive（非 ROI crop）**

- 数据：Trans10K **train 原图** + 帧级多类弱标签（若有）或 ROI 类名
- 目的：补 ROI 外 context；工程 **+2～3 天**

**P4F-7c — 换 backbone**

- `ViT-B-32` / `ViT-L-14`（仅当算力允许；§9 证 ViT-L **冻结 linear 更差**，需 **端到端 adapt**）
- 从 P3 流程 **重跑 P1→P3→P4**，工期 **1～2 周**

---

##### P4-full 文件清单

| 文件 | 职责 | 状态 |
|------|------|------|
| `train_contrastive_adapt.py` | contrastive + head-ft | ✅ 需扩展 patience/yaml |
| `eval_wise_ft_sweep.py` | WiSE-FT | ✅ |
| `wise_ft.py` | 插值工具 | ✅ |
| `configs/p4_contrastive_full.yaml` | 全量超参 | ⏳ 待建 |
| `scripts/run_p4_full.sh` | P4F-2～6 一键 | ⏳ 待建 |
| `outputs/.../p4_contrastive_full/` | 主实验产物 | ⏳ |

**总工期**：实现 **0.5 天** + 训练评测 **0.5～1 天** + ablation **0.5 天** → **1.5～2 天**。

**风险与回退**：

| 风险 | 现象 | 处理 |
|------|------|------|
| ep0 仍最佳 | val ep1 降 | patience=1；勿加 epoch |
| 全量仍 <77.04% | 无增益 | 停 P4；P5 结题 |
| SegMAN 不升 | GT 升、SegMAN 平 | 分割瓶颈；并行 P0 |
| 过拟合 hard 类 | door 升、cup 降 | 降 lr；减 epoch |

** realistic 区间**：P4-full 合理预期 GT **77.5%～78.5%**；**80% 仍不保证**。若 P4F-6 **<78%**，建议 **停止投入**、转方案 B。

---

### 5.7 P5 — 交付向：置信度拒识与按类策略（不提高 Top-1，提高可用性）

**目的**：在 Acc 暂不达 80% 时，保证 **抓取/demo 安全**。

**做法**：

1. 推理：`softmax(conf) < τ`（建议 τ=0.5～0.6）→ **拒识 / 人工确认**。
2. 报告 **coverage–accuracy 曲线**：如「覆盖 70% 样本时 Acc 82%」。
3. 强类（cup/eyeglass）自动抓取；weak 类（shelf/door）仅高置信执行。
4. 完成 `segment_and_classify.py` + `conf-threshold`。

**预期**：全局 Top-1 不变；**有效决策精度** 可显著高于 64.6%。  
**工期**：1～2 天。

**实测（2026-05-26，方案 B / `scripts/run_plan_b.sh`）**：

| 指标 | GT-ROI | SegMAN-ROI |
|------|--------|------------|
| 全局 Top-1 | 76.91% | 67.49% |
| @60% coverage Acc | **89.16%** ✅ | **80.67%** |
| @70% coverage Acc | **86.57%** ✅ | 77.51% |
| 按类拒识 Acc（coverage） | **86.08%**（71.05%） | **80.01%**（63.44%） |
| 全局 τ=0.5 拒识 Acc | 81.53%（86.83% cov） | 73.96%（81.60% cov） |

产物：`outputs/openclip_classifier/plan_b/`；结题摘要：`deliver_experiment_best/metrics_summary.md`。

## 6. 推荐实施路线

### 6.1 若目标仍是 **GT-ROI ≥ 80%**（研发）

```text
Week 1
  ✅ P1 T2 加深（4 block）→ GT 74.91%
  ✅ P2 层次分类 → GT 75.23% / SegMAN 67.09%（未达 77% stretch）
Week 2
  ✅ P3 hard mining + 轻量增强 → GT 76.91% / SegMAN 67.49%（闸门 PASS，未达 77% stretch）
  ✅ P3+P2 级联 ablation → GT 75.27%（**低于 P3 单头，不采用**）
  ✅ P4 快速验证 → WiSE-FT **76.94%** / contrastive **77.04%**（**未达 78%**）
  → 完整 P4 或 P5 拒识 + 结题
Week 3+
  P0-4c 分割边界 loss（并行，提 SegMAN-ROI）
  完整 P4（全量 contrastive + early stop）或接受 P3/contrastive 最佳 ~77%
```

** realistic 区间**：P4 快速验证最高 **77.04%**；距 80% **−2.96 pt**；轻量域适配 **不足以触达 78%**。

### 6.2 若目标为 **可交付 demo / 联调**（工程）

```text
保留 deliver_classifier_best.pth（74.88% / 64.61%）
  → P5 E2E + 置信度拒识
  → P0 分割弱类（提升 SegMAN-ROI 至 ~68%+）
  → 文档化双指标与未达 80% 原因（本文档 + 指南 §1.4）
```

---

## 7. 指标与验收建议（修订）

原单一指标 **GT-ROI ≥80%** 在现有证据下 **偏严**；建议改为 **组合验收**：


| 层级   | 指标                 | 当前            | 建议门槛                     |
| ---- | ------------------ | ------------- | ------------------------ |
| 分割   | mIoU               | 81.80%        | ≥80% ✅                   |
| 分类上界 | GT-ROI Acc         | T2 **74.88%** / P2 **75.23%** / **P3 76.91%** | ≥75%（阶段）✅ / ≥80%（stretch）❌ |
| 部署   | SegMAN-ROI Acc     | T2 **64.61%** / P2 **67.09%** / **P3 67.49%** | ≥65%（阶段）✅ / ≥70%（stretch） |
| 弱类   | door/wall F1 (GT)  | P3 door **66.61%** / wall **79.69%** | door ≥55% ✅ / wall ≥78% ✅ |
| 系统   | 高置信子集 Acc          | GT **89.16%** @60% cov / 按类拒识 **86.08%** | coverage≥60% 时 Acc≥78% ✅   |


---

## 8. 结论


| 问题               | 答案                                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **为什么没到 80%？**   | 11 类透明物体 **door/wall/window 结构混淆**占错分主体；**冻结/轻量微调 CLIP** 表征上限约 **75%**；**ROI 与分割** 进一步限制 weak 类；§9 已证 **更大模型/更紧 ROI 无效**。 |
| **5 pt 缺口能否补上？** | P4 快速验证最高 **77.04%**（+0.13 pt vs P3）；**78% / 80% 未达**；轻量 WiSE-FT/contrastive 不够，需完整 P4 或调整验收。 |
| **当前结果能否交付？**    | **分割 v2@6k 可交付**；**分类 P3 deliver**（GT **76.91%** / SegMAN **67.49%**）；80% 全局 stretch **未达标**；**方案 B 双指标结题 PASS**（高置信 **89.16%**）。 |
| **下一步最该做什么？**    | **结题/demo**：P3 + 按类拒识（`reject_thresholds_p3.json`）；**可选**：P0 弱类分割抬 SegMAN 全局 Acc。 |


---

## 9. 附录：关键文件索引


| 内容            | 路径                                                                               |
| ------------- | -------------------------------------------------------------------------------- |
| 交付 checkpoint | `outputs/openclip_classifier/deliver_classifier_best.pth`（**P3，76.91% / 67.49%**） |
| 交付 manifest   | `outputs/openclip_classifier/deliver_p3/deliver_manifest.json`              |
| T2 归档         | `outputs/openclip_classifier/deliver_classifier_t2_archived.pth`（74.88% / 64.61%） |
| T2 原 manifest  | `outputs/openclip_classifier/deliver_t2_best/deliver_manifest.json`           |
| **P1 checkpoint** | `outputs/openclip_classifier/p1_unfreeze4_noweight/best.pth`（含 encoder）     |
| P1 GT 评测      | `outputs/openclip_classifier/p1_unfreeze4_noweight/eval_gt_roi/summary.json`   |
| P1 SegMAN 评测  | `outputs/openclip_classifier/p1_unfreeze4_noweight/eval_segman_roi/summary.json` |
| P1 训练脚本     | `scripts/run_p1_train.sh`                                                          |
| **P2 Stage-1** | `outputs/openclip_classifier/p2_stage1_router/best.pth`                            |
| **P2 Stage-2** | `outputs/openclip_classifier/p2_stage2_structure/best.pth`                         |
| **P2 GT 评测** | `outputs/openclip_classifier/p2_eval_gt_roi/summary.json`（**75.23%**）            |
| **P2 SegMAN 评测** | `outputs/openclip_classifier/p2_eval_segman_roi/summary.json`（**67.09%**）    |
| P2 一键脚本     | `scripts/run_p2_1_build_hier_roi.sh` … `run_p2_4_eval_cascade.sh`                  |
| **P3 checkpoint** | `outputs/openclip_classifier/p3_p1_hardmining/best.pth`（含 encoder，**GT 76.91%**） |
| **P3 GT 评测** | `outputs/openclip_classifier/p3_p1_hardmining/eval_gt_roi/summary.json` |
| **P3 SegMAN 评测** | `outputs/openclip_classifier/p3_p1_hardmining/eval_segman_roi/summary.json`（**67.49%**） |
| P3 一键脚本     | `scripts/run_p3_train.sh`                                                          |
| **P3+P2 GT 评测** | `outputs/openclip_classifier/p3p2_eval_gt_roi/summary.json`（**75.27%**）        |
| **P3+P2 SegMAN 评测** | `outputs/openclip_classifier/p3p2_eval_segman_roi/summary.json`（**67.18%**） |
| P3+P2 一键脚本  | `scripts/run_p3p2_eval_cascade.sh`                                                 |
| **P4 WiSE-FT sweep** | `outputs/openclip_classifier/p4_wise_ft_sweep/sweep.json`（best α=0.95，**76.94%**） |
| **P4 contrastive** | `outputs/openclip_classifier/p4_contrastive_small/best.pth`（GT **77.04%**）     |
| P4 验证一键     | `scripts/run_p4_validate.sh`                                                       |
| P4 闸门         | `outputs/openclip_classifier/p4_validate_gate.json`                                |
| 完整归档          | `outputs/openclip_classifier/deliver_t2_best/`                                   |
| 指标 manifest   | `outputs/openclip_classifier/deliver_p3/deliver_manifest.json`              |
| 升级脚本        | `scripts/promote_p3_deliver.sh`                                             |
| T2 GT 评测      | `outputs/openclip_classifier/t2_unfreeze2_noweight/eval_gt_roi/summary.json`     |
| T2 SegMAN 评测  | `outputs/openclip_classifier/t2_unfreeze2_noweight/eval_segman_roi/summary.json` |
| 训练与复现指南       | `OpenCLIP_细分类训练与优化指南.md`                                                         |


---

## §5 实施进度（2026-05-26）

### P0

| 步骤 | 状态 | 说明 |
|------|------|------|
| P0-0 | ✅ 完成 | `eval_p0_baseline/`、`outputs/p0_weak_audit.md` |
| P0-1 | ✅ 训练完成 | 4000 iter；`best_mIoU_iter_2000.pth`（mIoU 81.00%） |
| P0-1-3 | ⚠️ 部分 | iter_2000 **未达** weak 三类 +1 pt |
| P0-2 | ✅ 完成 | 3443 ROIs → `data/trans10k_roi_segman_p0weak/val` |
| P0-3 | ❌ **未通过闸门** | SegMAN-ROI **61.81%**（基线 64.61%） |

**交付决策**：保留 **v2@6k** + **`deliver_classifier_best.pth`**。

### P1

| 步骤 | 状态 | 说明 |
|------|------|------|
| P1-0 | ✅ 完成 | `checkpoint_utils.py` encoder 存取 |
| P1-1 warmup | ✅ 完成 | `p1_warmup_unfreeze2/`，val 最高 73.69%，写出 encoder |
| P1-2 deepen | ✅ 完成 | `p1_unfreeze4_noweight/best.pth` @ epoch 1 |
| P1-3 闸门 | ✅ **通过** | GT **74.91%** / SegMAN **65.73%** |

详见 **§5.3** 完整数据表。

### P2

| 步骤 | 状态 | 说明 |
|------|------|------|
| P2-1 | ✅ 完成 | `trans10k_roi_gt_hier` / `trans10k_roi_segman_hier` |
| P2-2 | ⚠️ 部分 | 路由 Acc **95.91%**；object recall **90.4%**（闸门 92% 未达） |
| P2-3 | ✅ 完成 | 结构专头 wall **81.1%**；door 64.2% |
| P2-4 | ✅ 完成 | GT **75.23%** / SegMAN **67.09%** |
| P2-5 | ⚠️ 部分 | GT **75.23%** / SegMAN **67.09%**；未达 77% stretch |

详见 **§5.4「P2 实验结果汇总」** 及 P2-2～P2-5 分步实测表。

### P3

| 步骤 | 状态 | 说明 |
|------|------|------|
| P3-0 | ✅ 完成 | P1/P2 混淆 Top 对；door↔wall **434/433** |
| P3-1 | ✅ 完成 | `trans10k_roi_gt_p3`；door/wall/window 2× 权重 |
| P3-2 | ✅ 完成 | `roi_augment.py` + train `--aug p3` |
| P3-3 | ✅ 完成 | 从 P1 续训；early stop @ ep9；best ep5 |
| P3-4 | ✅ 完成 | GT **76.91%** / SegMAN **67.49%** |
| P3-5 | ✅ **PASS** | 超 P1 +2.00 pt；door F1 **66.61%**；未达 77% stretch |
| P3+P2 | ❌ **不采用** | GT **75.27%**（vs P3 单头 −1.64 pt）；略超 P2 +0.04 pt |

### P4（快速验证）

| 步骤 | 状态 | 说明 |
|------|------|------|
| P4-1 WiSE-FT | ✅ 完成 | best α=**0.95**，GT **76.94%**（+0.03 pt vs P3） |
| P4-2 contrastive | ✅ 完成 | 8000 ROI×4ep；best ep0 GT **77.04%**（+0.13 pt） |
| P4-3 闸门 78% | ❌ **未通过** | WiSE-FT / contrastive 均未达 **78%** |
| P4F-0～7 | ❌ **FAIL** | P4-full GT **76.91%**；未超 P4-small **77.04%** |

详见 **§5.6.1**、**§5.6.2「P4-full 实验结果汇总」**。

---

## 修订记录


| 日期         | 说明                                  |
| ---------- | ----------------------------------- |
| 2026-05-26 | v1.0：基于 §9 A/B/C 完整实验与 deliver 归档撰写 |
| 2026-05-26 | v1.1：§5.2 P0 扩展为完整实施步骤（P0-0～P0-4、命令、闸门、回退） |
| 2026-05-26 | v1.2：P0-0/1 已跑通；P0-2/3 脚本 `scripts/run_p0_remaining.sh` |
| 2026-05-26 | v1.3：P0-2/3 完成；闸门未通过，维持 v2@6k + T2 交付 |
| 2026-05-26 | v1.4：P1 两阶段完成；GT 74.91% / SegMAN 65.73%，闸门通过 |
| 2026-05-26 | v1.5：§5.3 写入 P1 完整实测；§5.4 扩展 P2 分步实施（P2-0～P2-5） |
| 2026-05-26 | v1.6：P2-1 完成；`trans10k_roi_gt_hier` / `trans10k_roi_segman_hier` |
| 2026-05-26 | v1.7：P2-2～5 完成；级联 GT 75.23% / SegMAN 67.09% |
| 2026-05-26 | v1.8：§5.4 写入 P2 完整实测汇总（指标/混淆/产物/交付结论） |
| 2026-05-26 | v1.9：§5.5 扩展 P3 完整实施步骤（P3-0～P3-5、命令、闸门、清单） |
| 2026-05-26 | v2.0：P3 全流程完成；GT **76.91%** / SegMAN **67.49%**；闸门 PASS |
| 2026-05-26 | v2.1：§5.1/§5.4/§6/§7/§8 同步 P3 实测；§5.5 步骤表填入 P3 结果 |
| 2026-05-26 | v2.2：P3+P2 级联 ablation；GT **75.27%**（低于 P3 单头，不采用）；§5.5 P3-4-4 实测 |
| 2026-05-26 | v2.3：P4 快速验证；WiSE-FT **76.94%** / contrastive **77.04%**；78% 闸门 FAIL |
| 2026-05-26 | v2.4：§5.6.2 方案 A 完整 P4 实施步骤（P4F-0～P4F-7） |
| 2026-05-26 | v2.5：P4-full 执行完成；GT **76.91%** FAIL；实验最高仍 **P4-small 77.04%** |
| 2026-05-26 | v2.6：**正式 deliver 升级为 P3**（76.91% / 67.49%）；T2 归档 |
| 2026-05-26 | v2.7：**方案 B 执行**；GT @60% cov **89.16%**；按类拒识 **86.08%**；§5.7/§7/§8 更新 |


