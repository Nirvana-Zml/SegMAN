# 路线 B：LASS + MMSCopE 实施清单

> **基线（已完成）**：`outputs/trans10k_segman_b/iter_80000.pth`，test **mIoU 80.71%**  
> **路线 B 最高 mIoU**：`outputs/trans10k_lass_mmscope_balanced10k/iter_10000.pth`，test **mIoU 81.76%**（见 §0.3）  
> **路线 B 正式权重（稳妥）**：`outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth`，test **mIoU 80.84%**（见 §0.1）；类均衡可选 **balanced10k/iter_6000**（↑6 类）  
> **基线 / fix5k / balanced10k 合一对比**：《路线B_基线_fix5k_balanced10k_对比分析.md》  
> **平衡微调说明**：**《路线B_平衡微调方案.md》**（10k，非 80k）  
> **目标**：mIoU **> 80.71%**，重点提升 **window / shelf / box**  
> **设计依据**：《透明物体分割_SegMAN优化设计说明书.md》§4～§5、§8；《项目实施步骤指南.md》B5～B7

---

## 0. 基线对照（80k @ iter_80000）


| 类别       | IoU (%)   | 路线 B 优先提升 |
| -------- | --------- | --------- |
| window   | 66.62     | ★★★       |
| shelf    | 67.61     | ★★★       |
| box      | 71.47     | ★★★       |
| freezer  | 73.90     | ★★        |
| door     | 75.04     | ★★        |
| **mIoU** | **80.71** | 总目标       |


详见：`Trans10K_SegMAN_B_训练与评测结果.md`

---

## 0.1 路线 B 正式结果（5k 微调，2026-05-20）

> **结论**：在修复 `ignore_index=255` 后，**5k iter 微调即可超过基线**；**不必**以 80k 为路线 B 交付条件。  
> **勿用**下列失败权重做评测或路线 C：`trans10k_lass_mmscope/iter_80000.pth`（mIoU 16.15%）、`trans10k_lass_mmscope_fix80k/`、`trans10k_lass_mmscope_fix80k_lr3e5/`（80k test 退化）。

### 训练设置（fix5k）

| 项 | 值 |
|----|-----|
| 配置 | `local_configs/segman_trans/segman_b_trans10k_lass.py` |
| 初始化 | `--load-from outputs/trans10k_segman_b/iter_80000.pth` |
| work-dir | `outputs/trans10k_lass_mmscope_fix5k` |
| max_iters | **5000** |
| optimizer.lr | **3e-5**（`--cfg-options`） |
| 验证 | `--no-validate`（训后单独 `test.py`） |

**代码修复（训前已合入）**：`lass_utils.semantic_seg_to_mbg`、`mmscope.semantic_seg_to_boundary` 及调用处均传入 `ignore_index=255`，避免 pad 被当作前景（此前 80k 无修复 run 塌缩为「全背景」的主因之一）。

**启动命令**：

```bash
mkdir -p outputs/trans10k_lass_mmscope_fix5k
python tools/train.py local_configs/segman_trans/segman_b_trans10k_lass.py \
  --work-dir outputs/trans10k_lass_mmscope_fix5k \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate \
  --cfg-options runner.max_iters=5000 data.workers_per_gpu=2 optimizer.lr=3e-5
```

### test 评测（Trans10K test，1000 张）

**权重**：`outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth`

```bash
python tools/test.py local_configs/segman_trans/segman_b_trans10k_lass.py \
  --checkpoint outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --eval mIoU
```

**评测说明**：与基线相同，均为 Trans10K **val 1000 张**（`data.test` → `img_dir/val`）；Δ = 路线 B 5k − 基线 80k。基线完整数字见 `Trans10K_SegMAN_B_训练与评测结果.md` §4～5。

### 总体指标对比

| 指标 | 基线 (80k) | 路线 B (5k) | Δ | 变化 |
|------|------------|-------------|-----|------|
| **mIoU** | 80.71 | **80.84** | **+0.13** | ↑ |
| aAcc | 96.07 | 95.92 | −0.15 | ↓（略） |
| mAcc | 88.14 | 87.45 | −0.69 | ↓（略） |

**小结**：总 **mIoU 超过基线**；aAcc / mAcc 略降，多与个别类 Acc↓、IoU↑ 的 trade-off 有关（如 window）。

### 各类 IoU 对比（12 类）

| 类别 | 基线 IoU | 5k IoU | Δ IoU | 变化 |
|------|----------|--------|-------|------|
| background | 96.71 | 96.37 | −0.34 | ↓ 略 |
| box | 71.47 | 71.44 | −0.03 | ≈ 持平 |
| bottle | 87.77 | 86.51 | −1.26 | ↓ |
| **window** | 66.62 | **76.27** | **+9.65** | **↑ 明显** |
| eyeglass | 92.85 | 90.81 | −2.04 | ↓ |
| freezer | 73.90 | 73.46 | −0.44 | ↓ 略 |
| jar_kettle | 84.04 | 82.06 | −1.98 | ↓ |
| door | 75.04 | 72.72 | −2.32 | ↓ |
| cup | 90.91 | 90.19 | −0.72 | ↓ 略 |
| wall | 82.72 | 82.77 | +0.05 | ≈ 持平 |
| bowl | 78.91 | 80.07 | +1.16 | ↑ |
| shelf | 67.61 | 67.44 | −0.17 | ≈ 持平 |
| **mIoU（均值）** | **80.71** | **80.84** | **+0.13** | **↑** |

**IoU 统计**：相对基线 **上升 3 类**（window、bowl、wall≈平）；**明显下降 4 类**（door、eyeglass、jar_kettle、bottle）；其余 **5 类** |Δ|≲0.5，视为持平。

### 各类 Acc 对比

| 类别 | 基线 Acc | 5k Acc | Δ Acc | 变化 |
|------|----------|--------|-------|------|
| background | 98.22 | 98.33 | +0.11 | ↑ 略 |
| box | 80.57 | 79.44 | −1.13 | ↓ |
| bottle | 89.93 | 87.97 | −1.96 | ↓ |
| window | 90.77 | 86.71 | −4.06 | ↓ |
| eyeglass | 95.92 | 96.04 | +0.12 | ↑ 略 |
| freezer | 77.31 | 76.07 | −1.24 | ↓ |
| jar_kettle | 88.32 | 87.53 | −0.79 | ↓ 略 |
| door | 83.98 | 82.58 | −1.40 | ↓ |
| cup | 96.55 | 95.39 | −1.16 | ↓ |
| wall | 91.99 | 91.41 | −0.58 | ↓ 略 |
| bowl | 85.51 | 86.98 | +1.47 | ↑ |
| shelf | 78.56 | 80.90 | +2.34 | ↑ |

**说明**：window 的 **IoU +9.65、Acc −4.06** 符合「区域划分更贴 GT、像素级 Acc 变严」的常见现象（基线亦存在 window Acc≫IoU）。

### 重点三类（阶段 3 目标类）

| 类别 | 基线 IoU | 5k IoU | Δ | 结论 |
|------|----------|--------|-----|------|
| **window** | 66.62 | **76.27** | **+9.65** | 达标，主论据 |
| shelf | 67.61 | 67.44 | −0.17 | 基本持平，未超基线 |
| box | 71.47 | 71.44 | −0.03 | 基本持平，未超基线 |

### 验收对照（阶段 3 目标）

| 指标 | 基线 | 目标 | 5k 实测 | 是否达标 |
|------|------|------|---------|----------|
| mIoU | 80.71% | > 80.71% | **80.84%** | ☑ |
| window IoU | 66.62% | 高于基线 | **76.27%** | ☑ |
| shelf IoU | 67.61% | 高于基线 | 67.44% | ✗（持平略低） |
| box IoU | 71.47% | 高于基线 | 71.44% | ✗（持平略低） |

**撰写建议（论文/答辩）**：主结论写 **mIoU 80.84%（+0.13）与 window +9.65 IoU**；shelf/box 如实写「与基线持平（±0.2 以内）」；door / eyeglass / bottle 等回落类可补 per-class 表或可视化。勿用 §0.2 中 80k 失败权重冒充路线 B 结果。

**说明**：创新/交付以 **总 mIoU + window** 为主论据；shelf/box 可配合 `vis_test` 或后续 10k 早停再挖。`load-from` 基线后继续 **全量 80k** 微调易退化，见 §0.2。

### 自检脚本

```bash
python scripts/verify_ignore_index_fix.py
python scripts/verify_mmscope_phase2.py
python scripts/check_ckpt_load.py   # 确认 test 时 missing=0
```

---

## 0.2 80k 实验记录（勿作正式权重）

| work-dir | lr | iter_80000 test mIoU | 说明 |
|----------|-----|----------------------|------|
| `trans10k_lass_mmscope` | 6e-5（默认） | **16.15%** | 含 ignore_index bug + 长训退化 |
| `trans10k_lass_mmscope_fix80k` | 6e-5 | **38.56%**（iter_4000 最高约 62.74%） | bug 已修，仍长训退化 |
| `trans10k_lass_mmscope_fix80k_lr3e5` | 3e-5 | **11.13%** | 与 5k 同 lr 但 80k 仍塌缩 |

**推荐**：路线 C / 论文主表优先 **`balanced10k/iter_10000.pth`**（mIoU **81.76%**，§0.3）；稳妥或类均衡对照可用 **`fix5k/iter_5000.pth`** 或 **`balanced10k/iter_6000.pth`**。勿用 §0.2 失败 80k 权重。

---

## 0.3 平衡微调结果（balanced10k，2026-05-23）

> **训练完成**：`outputs/trans10k_lass_mmscope_balanced10k`，**Iter 10000/10000**（配置 `segman_b_trans10k_lass_balanced.py`，**非 80k**）。  
> **完整表与命令**：**《路线B_平衡微调方案.md》§8**。

### 训练

| 项 | 值 |
|----|-----|
| 配置 | `segman_b_trans10k_lass_balanced.py` |
| 初始化 | `outputs/trans10k_segman_b/iter_80000.pth` |
| max_iters / lr | **10000** / **2e-5** |
| checkpoint | `iter_2000` … `iter_10000`（每 2000） |

### test 总览（val 1000 张）

| checkpoint | mIoU | Δ vs 基线 | ↑ / ≈ / ↓（\|Δ\|>0.2%） |
|------------|------|-----------|-------------------------|
| iter_6000 | 80.83% | +0.12 | **6 / 3 / 3** |
| iter_8000 | 81.59% | +0.88 | 5 / 2 / 5 |
| **iter_10000** | **81.76%** | **+1.05** | 5 / 2 / 5 |

对比：基线 **80.71%**；fix5k **80.84%**（§0.1）。

### iter_10000 重点类 vs 基线（IoU %）

| 类别 | 基线 | balanced 10k | Δ |
|------|------|--------------|-----|
| **window** | 66.62 | **82.91** | **+16.29** |
| shelf | 67.61 | 68.73 | +1.12 |
| box | 71.47 | 71.86 | +0.39 |
| bowl | 78.91 | 74.31 | **−4.60** |
| **mIoU** | 80.71 | **81.76** | **+1.05** |

**说明**：**bowl** 在 6k/8k/10k 均明显低于基线；**window** 为路线 B 最大增益。**iter_6000** 时 ↑ 类数最多（6 类），但 window 仅 +2.79。

**bowl 回落诊断（方案 3，已完成）**：对 `iter_10000` 做 vis + `scripts/analyze_bowl_confusion.py`；GT=bowl 像素上错分主要为 **background 9.29%**、**cup 6.19%**。详见 **《路线B_平衡微调方案.md》§8.6**。

**bowl 专项微调（方案 1，已完成）**：`bowl5k/iter_5000.pth`，mIoU **79.15%**，bowl **80.25%**；不宜作主推。见 **§10**。

**bowl 修补（方案 1b，已完成）**：`balanced_bowl3k_from10k` 未双达标；见 **§11.2–11.5**。

**balanced-v2（8k 扫完）**：终选 **4k**（mIoU 81.81 / bowl 80.82 / window 83.33）或 **6k**（shelf 67.73 / bowl 80.70）；**8k** mIoU 81.18、bowl 79.10、shelf 62.86 **不推荐**。见《路线B_平衡微调方案.md》**§12.7～12.8**、对比分析 **§5.9～5.12**。

### 权重选用建议

| 场景 | 权重 |
|------|------|
| 论文主表 **mIoU 最高** | `balanced10k/iter_10000.pth` |
| 多类 IoU 尽量不低于基线 | `balanced10k/iter_6000.pth` |
| 与 fix5k 接近、交付稳妥 | `fix5k/iter_5000.pth` |

逐类对比命令：`python scripts/compare_miou_vs_baseline.py outputs/trans10k_lass_mmscope_balanced10k/eval_iter_10000/eval_single_scale_*.json`

---

## 1. 总体架构（相对基线改动）

```text
输入图像
  → SegMANEncoder-LASS（Stage1~3：RSM → VSSM → LTAB）
  → SegMANDecoder-MMSCopE（BPM 边界图 + MSBEC 融合 + L_bd）
  → 12 类 mask（下游仍只用 mask，不用 logits 做细分类）
```

**与基线差异**


| 模块          | 基线                | 路线 B                       |
| ----------- | ----------------- | -------------------------- |
| Backbone    | `SegMANEncoder_b` | `SegMANEncoderLASS`        |
| Decode head | `SegMANDecoder`   | `SegMANDecoderMMSCopE`     |
| 额外损失        | CE only           | CE + **L_bd**（边界 BCE/Dice） |


---

## 2. 实施顺序（建议 2～3 周）

### 阶段 1：LASS 编码器（B5）— 约 1 周

> **本阶段只改编码器（Backbone）**，解码器仍用基线 `SegMANDecoder`。  
> **阶段 1 结束标志**：能 `build_backbone(SegMANEncoderLASS)` 前向通过，且 **20 iter 冒烟训练** loss 下降。  
> **不在此阶段追求 mIoU > 80.71**（要等阶段 2 MMSCopE + 阶段 3 完整训练）。

**数据流（相对基线 Attention）**

```text
v_r = Neighborhood Attention 输出
v   = dwconv(v_r) → SiLU → [RSM，需 M_bg] → VSSM/SS2D → [LTAB] → LayerNorm
x   = v + v_r   （再进 FFN）
```


| 子模块      | 作用（目的）                    | 针对的透明物体问题                  |
| -------- | ------------------------- | -------------------------- |
| **LTAB** | 低纹理区域加权，强化「几乎无纹理但仍属物体」的像素 | 大块透明平面、弱纹理的 window / shelf |
| **RSM**  | 在 SS2D 前抑制背景反射分量，保留物体微弱特征 | 反光、玻璃与背景混淆的 box / window   |


**RSM 的 `M_bg`（背景掩码）**：训练早期用 **GT** `M_bg = 1 - Dilate(前景)`；完整训练 B7-3 再改为 `gt+pred`（设计书 §4.3）。

### 阶段 2：MMSCopE 解码器（B6）— 约 1 周

> **本阶段只改编码器**：BPM + MSBEC + `L_bd`。冒烟分 **两步**：**2.6a** 基线 `SegMANEncoder_b`（先）→ **2.6b** `SegMANEncoderLASS_b`（后）。详见 **「阶段 2 详细手册」**。

### 阶段 3：完整训练（B7）— 约 1 周

> **LASS + MMSCopE 联合训练**；目标 test **mIoU > 80.71%**。**已达成**：fix5k **80.84%**（§0.1）；**balanced10k** **81.76%** @ iter_10000（§0.3）。80k 全量微调易退化（§0.2）。详见 **「阶段 3 详细手册」**。

---

## 阶段 1 详细手册（步骤 · 路径 · 命令 · 目的）

以下路径默认在 **Docker 容器内**；项目挂载为 `/workspace/segman`，训练命令在 `**segmentation`** 目录执行。


| 环境      | 项目根目录                   | 训练目录                                 |
| ------- | ----------------------- | ------------------------------------ |
| Docker  | `/workspace/segman`     | `/workspace/segman/segmentation`     |
| Windows | `D:\SegMAN-main\SegMAN` | `D:\SegMAN-main\SegMAN\segmentation` |


---

### 步骤 0：进入环境（每次新开终端先做）

**做什么**：进入已配置好的 conda + 工程目录。

**目的**：保证 `import mmseg`、CUDA、数据路径与基线训练一致。

```bash
docker exec -it segman_train bash
conda activate segman
cd /workspace/segman/segmentation
```

**验收**：`python -c "import mmseg; import torch; print(torch.cuda.is_available())"` 为 `True`。

---

### 步骤 1.1：实现并验证 LTAB 模块

**做什么**：实现「低纹理区域注意力」：`ltab.py` 根据特征图梯度估计纹理强弱，对 **低纹理区域** 提高权重，再乘回特征。

**目的**：透明物体（尤其 window、shelf）内部梯度弱，原版 VSSM 容易当背景；LTAB 让网络更关注「低纹理但仍属于物体」的区域。

**涉及文件**（已存在骨架，你可对照设计书 §4.2 微调）：

- `segmentation/mmseg/models/modules/ltab.py`
- `segmentation/mmseg/models/modules/__init__.py`

**命令（单元自检，不读数据、不训练）**：

```bash
cd /workspace/segman/segmentation

python -c "
import torch
from mmseg.models.modules.ltab import LTAB
x = torch.randn(2, 64, 64, 64)
y = LTAB(64)(x)
assert y.shape == x.shape, y.shape
print('1.1 LTAB ok')
"
```

**验收标准**：打印 `1.1 LTAB ok`，无报错。

**若失败**：检查是否在 `segmentation` 目录、`mmseg` 是否把 `models/modules` 加入 Python 路径（一般随 `import mmseg` 自动可用）。

---

### 步骤 1.2：实现并验证 RSM 模块

**做什么**：实现「反射抑制」：用背景掩码 `M_bg` 估计背景平均外观，从特征里减掉反射分量，再残差加回，输出送入 VSSM。

**目的**：透明场景里背景常透过物体形成 **反射高光**；在 SS2D 扫描前削弱背景反射，减轻 box/window 与背景混淆。

**涉及文件**：

- `segmentation/mmseg/models/modules/rsm.py`

**命令（单元自检）**：

```bash
python -c "
import torch
from mmseg.models.modules.rsm import ReflectionSuppression
x = torch.randn(2, 64, 64, 64)
m_bg = torch.zeros(2, 1, 64, 64)
m_bg[:, :, :20, :] = 1.0   # 上半部分假装背景
z = ReflectionSuppression(64)(x, m_bg)
assert z.shape == x.shape
print('1.2 RSM ok')
"
```

**验收标准**：打印 `1.2 RSM ok`。

**说明**：此步 **不要求** 真实 Trans10K mask，只验证「给定 `M_bg` 能算」；真实 `M_bg` 在步骤 1.5 训练时由 **GT 分割标注** 生成。

---

### 步骤 1.3：实现 `AttentionLASS`（改 Attention 前向）

**做什么**：复制/继承 `segman_encoder.py` 里的 `Attention`，在 `**global_mode=False`** 分支中插入 RSM 与 LTAB，顺序为：

```text
v_r → dwconv → SiLU → RSM(v, m_bg) → VSSM → LTAB → norm → + v_r
```

`Attention.forward` 需增加可选参数，例如 `m_bg=None`；`m_bg is None` 时 **跳过 RSM**（便于步骤 1.4 先测 LTAB-only 或调试）。

**目的**：把 1.1、1.2 的两个子模块接到 SegMAN 原有 NA+VSSM 路径上，**不改动 Stage4 全局注意力**。

**涉及文件（需新建）**：

- `segmentation/mmseg/models/backbones/segman_encoder_lass.py`  
  - 类：`AttentionLASS`（改 `forward`）  
  - 后续同文件内：`BlockLASS`、`BasicLayerLASS`（让 Block 把 `m_bg` 传给 `AttentionLASS`）

**建议实现顺序（写代码时）**：

1. 只加 **LTAB**（`m_bg=None`），跑通 1.4 前向。
2. 再加 **RSM**，`forward(x, pos_enc, m_bg=None)`。

**本地编辑位置参考**（基线）：

```559:604:segmentation/mmseg/models/backbones/segman_encoder.py
        if not self.global_mode:
            ...
            v = self.dwconv(v_r)
            v = F.silu(v)
            v = self.ssm(v)
            v = self.norm(v.reshape(B, -1, H, W).contiguous())
            x = v + v_r
```

在 `ssm` 前插入 RSM，在 `norm` 前插入 LTAB。

**验收（写完 AttentionLASS 后，仍可不注册 BACKBONES）**：

```bash
python -c "
import torch
from mmseg.models.backbones.segman_encoder_lass import AttentionLASS
m = AttentionLASS(96, 4, 7, 1, False, None, True, 1)
x = torch.randn(1, 96, 64, 64)
pos = (torch.randn(1,64,64,8), torch.randn(1,64,64,8))
m_bg = torch.zeros(1, 1, 64, 64)
y = m(x, pos, m_bg=m_bg)
print('1.3 AttentionLASS ok', y.shape)
"
```

（若类名/构造参数与基线 `Attention` 一致，按你实际签名改上面脚本。）

---

### 步骤 1.4：实现 `SegMANEncoderLASS` 并注册到 MMSeg

**做什么**：

1. 在 `segman_encoder_lass.py` 中定义 `SegMANEncoderLASS`（可复制 `SegMANEncoder` + `SegMANEncoder_b` 注册方式）。
2. `Block` 使用 `token_mixer=AttentionLASS`，`forward` 增加 `seg_map` 或 `m_bg` 参数并向下传递。
3. `SegMANEncoderLASS.forward(self, x, seg_map=None)`：
  - 若 `seg_map` 为 GT 语义标签 `B×H×W`：  
   `fg = (seg_map > 0).float()` → 下采样到各 stage 分辨率 → `m_bg = 1 - dilate(fg)`。  
  - Stage 0～2 的 block 传入对应分辨率的 `m_bg`；Stage 3 保持 `global_mode=True`，不用 LASS。
4. 在 `segmentation/mmseg/models/backbones/__init__.py` 增加：
  ```python
   from .segman_encoder_lass import SegMANEncoderLASS
   # __all__ 中加入 'SegMANEncoderLASS'
  ```
5. 在 `segmentation/mmseg/models/backbones/segman_encoder_lass.py` 末尾：
  ```python
   @BACKBONES.register_module()
   def SegMANEncoderLASS_b(pretrained=None, pretrained_cfg=None, **args):
       ...
  ```

**目的**：让 MMSeg 的 `build_backbone(cfg)` 能构建「带 LASS 的 SegMAN-B」，输出 **4 个尺度特征图**，形状与基线一致，以便原 `SegMANDecoder` 直接对接。

**命令（构建 backbone + 前向）**：

```bash
cd /workspace/segman/segmentation

python -c "
from mmseg.models import build_backbone
import torch
cfg = dict(
    type='SegMANEncoderLASS_b',
    pretrained=None,
    style='pytorch',
    lass_cfg=dict(enable_stages=[0, 1, 2]))
model = build_backbone(cfg)
model.eval()
x = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    outs = model(x)
print('1.4 shapes:', [o.shape for o in outs])
"
```

**验收标准**：

- 无 import / shape 报错。  
- 输出为 **4 个 tensor**，通道大致为 `[96, 160, 364, 560]`（与基线 SegMAN-B 一致），空间尺寸随 stage 递减。

**可选：加载 ImageNet 预训练（与基线相同）**：

```bash
python -c "
from mmseg.models import build_backbone
cfg = dict(
    type='SegMANEncoderLASS_b',
    pretrained='/workspace/segman/pretrained/SegMAN_Encoder_b.pth.tar',
    style='pytorch')
model = build_backbone(cfg)
print('1.4 pretrained load ok')
"
```

**说明**：`strict=False` 加载时，**LTAB/RSM 新增参数**会随机初始化，属正常；与基线共用的 NA/VSSM 权重应能对上。

---

### 步骤 1.5：编写「仅换 backbone」的配置 + 冒烟训练

**做什么**：

1. 新建 `local_configs/segman_trans/segman_b_trans10k_lass_enc_only.py`：
  - 继承 `segman_b_trans10k.py`  
  - `model.backbone.type='SegMANEncoderLASS_b'`（及 `lass_cfg`）  
  - `**decode_head` 仍为 `SegMANDecoder`**（阶段 2 再换 MMSCopE）
2. 跑 **20 iter** 冒烟，确认数据 pipeline、loss、反传正常。

**目的**：在 **不接 MMSCopE、不跑满 80k** 的情况下，验证「LASS 编码器 + 原解码器 + Trans10K 数据」整条训练链路能跑通；避免一次改太多难以排错。

**命令**：

```bash
cd /workspace/segman/segmentation
conda activate segman

# 先打印配置，确认 backbone 类型
python tools/print_config.py \
  local_configs/segman_trans/segman_b_trans10k_lass_enc_only.py

# 冒烟 20 iter（可不从 80k 加载，仅验证能训）
python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_lass_enc_only.py \
  --work-dir outputs/trans10k_lass_enc_smoke \
  --cfg-options runner.max_iters=20 data.workers_per_gpu=2

# 推荐：从基线 80k 加载，看 loss 是否合理（需 config 或代码支持 load_from）
python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_lass_enc_only.py \
  --work-dir outputs/trans10k_lass_enc_smoke \
  --cfg-options runner.max_iters=20 data.workers_per_gpu=2 \
  --load-from outputs/trans10k_segman_b/iter_80000.pth
```

**验收标准**：

- 不出现 shape / CUDA / BatchNorm 报错。  
- 日志里 `Iter [20/20]` 出现，`loss` 为有限数值（不要求 mIoU）。  
- 若 `--load-from iter_80000.pth`：初始 loss 应 **低于** 随机初始化很多。

**注意**：若训练时 RSM 需要 `seg_map`，须在 `EncoderDecoder` 或自定义 backbone 调用里把 **batch 的 `gt_semantic_seg`** 传给 `SegMANEncoderLASS.forward`；这是 1.4 写代码时的关键集成点（MMSeg 默认只把 image 送 backbone）。

---

### 步骤 1.6（可选）：仅 LTAB 或仅 RSM 消融配置

**做什么**：在 `lass_cfg` 里设 `enable_rsm=False` 或 `enable_ltab=False`，各跑短训/评测。

**目的**：写论文或报告时的 **消融实验**，证明两个子模块各自贡献。

**命令**：与 1.5 相同，仅改 config 中 `lass_cfg`，`max_iters` 可 500～2000。

---

### 阶段 1 完成检查表


| 步骤                      | 状态  | 验收                        |
| ----------------------- | --- | ------------------------- |
| 0 环境                    | ☐   | `import mmseg`、GPU 可用     |
| 1.1 LTAB                | ☑   | `1.1 LTAB ok`             |
| 1.2 RSM                 | ☑   | `1.2 RSM ok`              |
| 1.3 AttentionLASS       | ☑   | 单模块前向 ok                  |
| 1.4 SegMANEncoderLASS_b | ☑   | 4 尺度特征 shape 正确           |
| 1.5 冒烟 20 iter          | ☑   | `trans10k_lass_enc_smoke` |


**阶段 1 通过后** → 进入 **阶段 2：MMSCopE 解码器（B6）**（见下文「阶段 2 详细手册」）。

---

### 阶段 1 与基线文件对照（改哪里）


| 基线文件                                              | 阶段 1 动作                                              |
| ------------------------------------------------- | ---------------------------------------------------- |
| `backbones/segman_encoder.py`                     | **不直接改**；逻辑抄到 `segman_encoder_lass.py`               |
| `modules/ltab.py`, `rsm.py`                       | 新建（已有）                                               |
| `backbones/__init__.py`                           | 增加 `SegMANEncoderLASS` 导出                            |
| `local_configs/segman_trans/segman_b_trans10k.py` | 复制为 `segman_b_trans10k_lass_enc_only.py`，只改 backbone |
| `outputs/trans10k_segman_b/iter_80000.pth`        | 1.5 可选 `--load-from`，迁移权重                            |


---

## 阶段 2 详细手册（步骤 · 路径 · 命令 · 目的）

> **本阶段只改编码器侧**：BPM + MSBEC + `L_bd`（backbone 在冒烟阶段分两步，见下）。  
> **阶段 2 结束标志**：`build_head(SegMANDecoderMMSCopE)` 前向通过；`loss_seg + loss_bd` 可反传；**2.6a + 2.6b 两次 20 iter 冒烟** 均无崩溃。  
> **不在此阶段追求 mIoU > 80.71%**（阶段 3：LASS + MMSCopE 完整 80k）。

### 冒烟策略（推荐顺序，必读）

按下面顺序做两次短训，变量最少、最接近最终路线 B：


| 顺序       | 步骤   | Backbone                   | Segmentor            | 配置文件                                         | 目的                                                                    |
| -------- | ---- | -------------------------- | -------------------- | -------------------------------------------- | --------------------------------------------------------------------- |
| **① 先做** | 2.6a | 基线 `SegMANEncoder_b`       | `EncoderDecoder`     | `segman_b_trans10k_mmscope_dec_only.py`      | **只验证 MMSCopE**：BPM/MSBEC/L_bd、shape、反传；`--load-from iter_80000` 匹配最好 |
| **② 再做** | 2.6b | 阶段 1 `SegMANEncoderLASS_b` | `EncoderDecoderLASS` | `segman_b_trans10k_lass_mmscope_dec_only.py` | **联合验收**：LASS + MMSCopE 接口（含 `gt_semantic_seg`→`M_bg`）正常，再进阶段 3       |


```text
2.1～2.5 实现 mmscope + SegMANDecoderMMSCopE
    ↓
2.6a  基线 backbone + MMSCopE 解码器（必做，优先）
    ↓ 通过后再做
2.6b  LASS backbone + MMSCopE 解码器（必做，进阶段 3 前）
    ↓
阶段 3  segman_b_trans10k_lass.py 完整 80k
```

**说明**：阶段 1 已冒烟过 LASS（`trans10k_lass_enc_smoke`），2.6a 不必重复证明 encoder；2.6b 不能省略，否则阶段 3 一上来 80k 易因 encoder/decoder 联调问题难排查。

**数据流（相对基线 `SegMANDecoder`）**

```text
多尺度特征 c1～c4
  → forward_mlp_decoder → _c, _c2, _c3, _c4
  → BPM → P_bd, W_bd；对 _c2/_c 做边界加权
  → forward_winssm → F_sem
  → MSBEC(P_bd, 多尺度) → F_ref；F_fuse = F_sem + MSBEC 融合
  → cls_seg → 12 类 logits
训练额外：Y_bd = dilate(fg) - erode(fg) → L_bd(BCE/Dice)
```


| 子模块       | 作用（目的）                       | 针对的透明物体问题                       |
| --------- | ---------------------------- | ------------------------------- |
| **BPM**   | 预测边界概率 `P_bd` 并生成空间权重 `W_bd` | 边界模糊、轮廓不准（window / shelf / box） |
| **MSBEC** | 多尺度边界卷积 + 与 `F_sem` 融合       | 轮廓附近语义增强                        |
| **L_bd**  | 边界 BCE/Dice 监督               | 显式优化边界质量                        |


路径与环境约定与阶段 1 相同（Docker：`/workspace/segman/segmentation`）。

---

### 步骤 2.0：进入环境（每次新开终端先做）

**做什么**：进入 Docker + conda + `segmentation` 目录。

**目的**：与阶段 1、基线训练环境一致。

```bash
docker exec -it segman_train bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate segman
cd /workspace/segman/segmentation

# 若 mmcv 报 TabError / torch.amp，新容器执行一次
python /workspace/segman/scripts/fix_mmcv_torch21.py
```

**验收**：

```bash
python -c "import mmseg; import torch; print('cuda:', torch.cuda.is_available())"
```

---

### 步骤 2.1：实现边界概率模块 BPM

**做什么**：新建 `segmentation/mmseg/models/modules/mmscope.py`，实现 `BoundaryProbabilityModule`（设计书 §5.1）：

```text
Conv3×3 → BN → ReLU → Conv3×3 → Conv1×1 → Sigmoid  →  P_bd  (B×1×H×W)
由 P_bd 再生成边界注意力权重 W_bd（如 1×1 Conv + Sigmoid on Concat(P_bd, Pool(P_bd))）
```

**目的**：得到可学习的边界概率图，并生成空间权重，用于调制解码特征。

**涉及文件**：

- `mmseg/models/modules/mmscope.py`（新建）
- `mmseg/models/modules/__init__.py`（导出子模块）

**命令（单元自检，CPU 即可）**：

```bash
cd /workspace/segman/segmentation

python -c "
import torch
from mmseg.models.modules.mmscope import BoundaryProbabilityModule
m = BoundaryProbabilityModule(in_channels=180)
x = torch.randn(2, 180, 64, 64)
p_bd, w_bd = m(x)
assert p_bd.shape == (2, 1, 64, 64), p_bd.shape
assert w_bd.shape == (2, 1, 64, 64), w_bd.shape
print('2.1 BPM ok', p_bd.shape, w_bd.shape)
"
```

**验收标准**：打印 `2.1 BPM ok`，`P_bd`、`W_bd` 均为 `B×1×H×W`。

---

### 步骤 2.2：实现多尺度边界增强 MSBEC

**做什么**：在同一文件实现 `MultiScaleBoundaryEnhance`（设计书 §5.2）：


| 分支  | 分辨率 | 操作                  |
| --- | --- | ------------------- |
| S0  | H×W | DWConv3×3 + Conv1×1 |
| S1  | H/2 | stride=2 + DWConv   |
| S2  | H/4 | stride=4 + DWConv   |


```text
F_ref = Conv( Concat( Upsample(S0,S1,S2), P_bd ) )
F_fuse = Conv( Concat(F_sem, F_ref) ) + F_sem
```

**目的**：在轮廓附近融合多尺度边界特征。

**命令（前向 + 反传，需 GPU）**：

```bash
python -c "
import torch
from mmseg.models.modules.mmscope import MultiScaleBoundaryEnhance
m = MultiScaleBoundaryEnhance(channels=180).cuda()
f_sem = torch.randn(2, 180, 64, 64, requires_grad=True, device='cuda')
p_bd = torch.rand(2, 1, 64, 64, device='cuda')
out = m(f_sem, p_bd)
loss = out.sum()
loss.backward()
assert out.shape == f_sem.shape
print('2.2 MSBEC ok', out.shape)
"
```

**验收标准**：`2.2 MSBEC ok`，`loss.backward()` 无 shape 错误。

---

### 步骤 2.3：实现 `SegMANDecoderMMSCopE` 并注册

**做什么**：

1. 新建 `segmentation/mmseg/models/decode_heads/segman_decoder_mmscope.py`，**继承** `SegMANDecoder`。
2. 在 `__init__` 中根据 `mmscope_cfg` 构建 BPM、MSBEC。
3. **改写 `forward`**（基线插入点参考 `segman_decoder.py`）：

```text
x, c2, c3, c4 = forward_mlp_decoder(...)
p_bd, w_bd = BPM(x)
c2', x' = c2 * (1 + η·W_bd), x * (1 + η·W_bd)   # 设计书对 _c2、_c 加权
x = forward_winssm(x', c2', c3, c4)              # F_sem
x = MSBEC(x, p_bd)                               # 边界多尺度融合
logits = cls_seg(x)
```

1. `@HEADS.register_module()` 注册 `SegMANDecoderMMSCopE`。
2. 在 `decode_heads/__init__.py` 增加 import 与 `__all__`。

**目的**：MMSeg 能 `build_head`，且 `in_channels=[96,160,364,560]`、`channels=180` 与基线一致。

**命令（mock 多尺度特征，需 GPU）**：

```bash
python -c "
import torch
from mmseg.models import build_head
cfg = dict(
    type='SegMANDecoderMMSCopE',
    in_channels=[96, 160, 364, 560],
    in_index=[0, 1, 2, 3],
    channels=180,
    feat_proj_dim=320,
    dropout_ratio=0.1,
    num_classes=12,
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    align_corners=False,
    mmscope_cfg=dict(boundary_loss_weight=0.4, refine_eta=0.1),
    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
)
head = build_head(cfg).cuda().eval()
feats = [
    torch.randn(1, 96, 128, 128, device='cuda'),
    torch.randn(1, 160, 64, 64, device='cuda'),
    torch.randn(1, 364, 32, 32, device='cuda'),
    torch.randn(1, 560, 16, 16, device='cuda'),
]
with torch.no_grad():
    out = head(feats)
print('2.3 decoder ok', out.shape)
"
```

**验收标准**：`out` 为 `B×12×H×W`（通常与 `c2` 同分辨率，512 输入下约 128×128）。

---

### 步骤 2.4：边界 GT `Y_bd` 生成

**做什么**：实现 `semantic_seg_to_boundary(seg_map, dilate_k=3, erode_k=3)`（可放在 `mmscope.py`）：

```text
fg = (seg_map > 0).float()
Y_bd = Dilate(fg, k) - Erode(fg, k)
```

**目的**：训练边界监督；在 head 的 `losses` / `forward_train` 中由 `gt_semantic_seg` 即时生成，**不必改 data pipeline**。

**命令**：

```bash
python -c "
import torch
from mmseg.models.modules.mmscope import semantic_seg_to_boundary
y = torch.zeros(2, 1, 64, 64, dtype=torch.long)
y[:, :, 20:40, 20:40] = 3
bd = semantic_seg_to_boundary(y, dilate_k=5, erode_k=5)
assert bd.shape == (2, 1, 64, 64)
print('2.4 Y_bd ok', bd.sum().item())
"
```

**验收标准**：打印 `2.4 Y_bd ok`，轮廓带有非零边界像素。

---

### 步骤 2.5：增加 `loss_bd` 与配置项

**做什么**：

1. 在 `SegMANDecoderMMSCopE` 中重写 `forward_train` / `losses`：
  - `loss_seg`：基类 CE（`loss_decode`）
  - `loss_bd`：`BCEWithLogits` 或 `Dice` on `P_bd` vs `Y_bd`
  - `loss = loss_seg + λ_bd * loss_bd`（`λ_bd` = `mmscope_cfg.boundary_loss_weight`，建议 **0.2～0.5**，默认 **0.4**）
2. 新建 **两个** 配置文件（对应 2.6a / 2.6b，均只改 `decode_head`，`runner.max_iters` 冒烟时可改为 20）：


| 配置文件                                         | 用于步骤         | Backbone                           | `model.type`         |
| -------------------------------------------- | ------------ | ---------------------------------- | -------------------- |
| `segman_b_trans10k_mmscope_dec_only.py`      | **2.6a（先做）** | `SegMANEncoder_b`                  | `EncoderDecoder`     |
| `segman_b_trans10k_lass_mmscope_dec_only.py` | **2.6b（后做）** | `SegMANEncoderLASS_b` + `lass_cfg` | `EncoderDecoderLASS` |


**2.6a 配置示例**（只换解码器）：

```python
_base_ = ['./segman_b_trans10k.py']
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='SegMANEncoder_b', ...),  # 与基线相同，勿改 LASS
    decode_head=dict(
        type='SegMANDecoderMMSCopE',
        mmscope_cfg=dict(
            boundary_loss_weight=0.4,
            refine_eta=0.1,
            dilate_kernel=5,
            erode_kernel=5,
        ),
    ),
)
```

**2.6b 配置示例**（LASS + MMSCopE）：

```python
_base_ = ['./segman_b_trans10k_lass_enc_only.py']  # 或合并 lass_enc_only 的 backbone 段
model = dict(
    type='EncoderDecoderLASS',
    backbone=dict(type='SegMANEncoderLASS_b', lass_cfg=dict(...)),
    decode_head=dict(
        type='SegMANDecoderMMSCopE',
        mmscope_cfg=dict(boundary_loss_weight=0.4, refine_eta=0.1, ...),
    ),
)
```

**目的**：日志可见 `loss_seg` 与 `loss_bd`；两套 config 分别服务「只验解码器」与「验 LASS 联调」。

**命令**：

```bash
python tools/print_config.py local_configs/segman_trans/segman_b_trans10k_mmscope_dec_only.py
python tools/print_config.py local_configs/segman_trans/segman_b_trans10k_lass_mmscope_dec_only.py
```

**验收标准**：两份配置里 `decode_head.type` 均为 `SegMANDecoderMMSCopE`；2.6b 还须为 `SegMANEncoderLASS_b` + `EncoderDecoderLASS`。

---

### 步骤 2.6a：冒烟训练 — 只换解码器（20 iter，**必做 · 优先**）

**做什么**：`SegMANEncoder_b` + `SegMANDecoderMMSCopE`，短训 20 iter；**不跑验证**（`--no-validate`）。

**目的**：在 **不动 LASS** 的前提下，单独验证 BPM / MSBEC / `L_bd` 与数据 pipeline；出问题可断定在 decode head。`--load-from iter_80000.pth` 与 backbone/原 decoder 权重对齐最好。

**命令**：

```bash
cd /workspace/segman/segmentation
conda activate segman

python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_mmscope_dec_only.py \
  --work-dir outputs/trans10k_mmscope_dec_smoke \
  --cfg-options runner.max_iters=20 data.workers_per_gpu=2 \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate
```

**验收标准**：

- 无 shape / CUDA / SyncBN 报错；
- `Iter [20/20]`，`loss` / `loss_bd`（或等价 key）为有限数值；
- `outputs/trans10k_mmscope_dec_smoke/iter_20.pth` 已保存。

**说明**：BPM/MSBEC 为新层，`--load-from` 时 missing keys 正常；初始 `loss` 应明显低于随机初始化。

**未通过 2.6a 前不要进行 2.6b。**

---

### 步骤 2.6b：冒烟训练 — LASS + MMSCopE（20 iter，**必做 · 进阶段 3 前**）

**做什么**：`SegMANEncoderLASS_b` + `EncoderDecoderLASS` + `SegMANDecoderMMSCopE`，短训 20 iter。

**目的**：验证阶段 1 编码器与阶段 2 解码器 **一起** 训练无接口问题（`gt_semantic_seg`→RSM 的 `M_bg`、四尺度特征对接 decode head）。**不是**重复阶段 1 的 encoder-only 冒烟。

**命令**：

```bash
python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_lass_mmscope_dec_only.py \
  --work-dir outputs/trans10k_lass_mmscope_dec_smoke \
  --cfg-options runner.max_iters=20 data.workers_per_gpu=2 \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate
```

**验收标准**：同 2.6a；checkpoint 在 `outputs/trans10k_lass_mmscope_dec_smoke/`。

**说明**：`--load-from` 时除 BPM/MSBEC 外，还会有 LTAB/RSM 的 missing keys（随机初始化），属正常。

**阶段 2 完成条件**：2.6a **与** 2.6b **均** 通过。

---

### 步骤 2.7（可选）：消融配置

**做什么**：在 `mmscope_cfg` 中设 `enable_bpm=False` / `enable_msbec=False` / `boundary_loss_weight=0`，各跑 500～2000 iter。

**目的**：论文/报告消融（仅 BPM / 仅 MSBEC / 无边界 loss）。

**命令**：建议基于 **2.6a 配置**（基线 backbone）做消融，减少 LASS 变量；`max_iters=500～2000`。

---

### 阶段 2 完成检查表


| 步骤                         | 状态  | 验收                                            |
| -------------------------- | --- | --------------------------------------------- |
| 2.0 环境                     | ☑   | `cuda: True`                                  |
| 2.1 BPM                    | ☑   | `P_bd` 为 B×1×H×W                              |
| 2.2 MSBEC                  | ☑   | GPU 上前向+反传 OK                                 |
| 2.3 SegMANDecoderMMSCopE   | ☑   | mock 特征 → logits shape OK                     |
| 2.4 Y_bd                   | ☑   | `2.4 Y_bd ok`                                 |
| 2.5 loss_bd + 双 config     | ☑   | 两份 config 已建                                  |
| **2.6a** 基线 backbone 冒烟    | ☑   | `trans10k_mmscope_dec_smoke/iter_20.pth`      |
| **2.6b** LASS + MMSCopE 冒烟 | ☑   | `trans10k_lass_mmscope_dec_smoke/iter_20.pth` |
| 2.7 消融（可选）                 | ☐   | —                                             |


**阶段 2 通过后**（2.6a **且** 2.6b）→ 进入 **阶段 3：完整训练（B7）**（见下文「阶段 3 详细手册」）。

---

### 阶段 2 与基线文件对照（改哪里）


| 基线文件                                       | 阶段 2 动作                                                        |
| ------------------------------------------ | -------------------------------------------------------------- |
| `decode_heads/segman_decoder.py`           | **不直接改**；逻辑扩展到 `segman_decoder_mmscope.py`                     |
| `modules/mmscope.py`                       | **新建** BPM + MSBEC + `semantic_seg_to_boundary`                |
| `decode_heads/__init__.py`                 | 导出 `SegMANDecoderMMSCopE`                                      |
| `modules/__init__.py`                      | 导出 mmscope 子模块                                                 |
| `segman_b_trans10k.py`                     | 复制为 `*_mmscope_dec_only.py`；阶段 3 用 `segman_b_trans10k_lass.py` |
| `outputs/trans10k_segman_b/iter_80000.pth` | 2.6a / 2.6b 均推荐 `--load-from`                                  |
| `outputs/trans10k_lass_enc_smoke/`         | 阶段 1 已完成；**不能代替** 2.6b                                         |


---

### 阶段 2：MMSCopE 解码器（B6）— 约 1 周（总览）


| 步骤       | 任务                     | 产出文件                                     | 验收                            |
| -------- | ---------------------- | ---------------------------------------- | ----------------------------- |
| 2.1      | 边界概率 BPM               | `mmseg/models/modules/mmscope.py`        | `P_bd` 为 `B×1×H×W`            |
| 2.2      | MSBEC 多尺度融合            | 同上                                       | `forward` + `loss.backward()` |
| 2.3      | `SegMANDecoderMMSCopE` | `decode_heads/segman_decoder_mmscope.py` | 注册到 `HEADS`                   |
| 2.4      | 边界 GT `Y_bd`           | head 内或 `mmscope.py`                     | 与 mask 对齐                     |
| 2.5      | `loss_bd` + 双 config   | `mmscope_cfg`                            | 总 loss 可反传                    |
| **2.6a** | 基线 backbone 冒烟         | `trans10k_mmscope_dec_smoke`             | **优先**；只验 MMSCopE             |
| **2.6b** | LASS + MMSCopE 冒烟      | `trans10k_lass_mmscope_dec_smoke`        | 进阶段 3 前必做                     |
| 2.7      | 消融（可选）                 | config 开关                                | 建议用 2.6a 配置                   |


**解码器插入点**：`forward_mlp_decoder` 得到 `_c2,_c` 后乘 `W_bd(P_bd)`；`forward_winssm` 后做 MSBEC 与 `F_sem` 融合。

---

## 阶段 3 详细手册（步骤 · 路径 · 命令 · 目的）

> **前提**：阶段 1（LASS）与阶段 2（MMSCopE，含 **2.6a + 2.6b**）均已完成。  
> **本阶段目标**：在 Trans10K-v2 上验证 `EncoderDecoderLASS` + `SegMANDecoderMMSCopE`，test **mIoU > 80.71%**（基线 80.71%），并重点提升 **window / shelf / box**。  
> **已达成（推荐交付）**：fix5k **80.84%**（§0.1）；**balanced10k** **81.76%** @ `iter_10000.pth`（§0.3）。  
> **历史 80k 目录**：`outputs/trans10k_lass_mmscope/` 等 **勿作正式权重**（见 **§0.2**）。

### 训练策略说明（两种做法）

| 做法 | 适用 | 说明 |
|------|------|------|
| **5k 微调（推荐交付，§0.1）** | 创新项目、在基线 ckpt 上插 LASS+MMSCopE | `max_iters=5000`，`optimizer.lr=3e-5`，`--load-from iter_80000.pth`；**已达标 mIoU** |
| **方案 A：一键 80k** | 与基线 iter 数对齐（可选） | 实测易退化，仅作对照实验；见 §0.2 |
| **方案 B：设计书三阶段 B7-1→2→3** | 追求更稳收敛、分阶段调参 | 40k + 40k 链式 `resume`；见步骤 3.4～3.6 |

**损失（当前实现）**：`L_total = L_seg + λ_bd·L_bd`（`λ_bd` 默认 0.4）；RSM 的 `M_bg` 由 **GT** 生成（`EncoderDecoderLASS`）。  
**设计书进阶**：`bg_mask_mode='gt+pred'`、冻结 Stage4 等若 config 未单独实现，可按步骤 3.4 注释用 `paramwise_cfg` / 后续扩展 `lass_cfg` 完成；**方案 A 不依赖这两项也可开训**。

**权重加载关系**

```text
ImageNet SegMAN_Encoder_b.pth.tar  →  backbone.init_weights（config.pretrained）
iter_80000.pth（基线 80k）         →  --load-from（迁移 decoder 主体 + 共有 encoder 层）
LTAB / RSM / BPM / MSBEC           →  新参数，load-from 时 missing keys，随机初始化（正常）
```

---

### 步骤 3.0：进入环境 + 确认前置产物

**做什么**：与阶段 1/2 相同进入 Docker；确认数据、基线权重、阶段 2 冒烟目录存在。

**目的**：避免在错误目录或缺数据情况下启动长达数天的 80k 训练。

```bash
docker exec -it segman_train bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate segman
cd /workspace/segman/segmentation

python /workspace/segman/scripts/fix_mmcv_torch21.py   # 新容器执行一次

# 数据
ls data/trans10k/img_dir/train | head
# 基线 80k
ls -lh outputs/trans10k_segman_b/iter_80000.pth
# 阶段 2 已通过（可选复查）
ls outputs/trans10k_mmscope_dec_smoke/iter_20.pth
ls outputs/trans10k_lass_mmscope_dec_smoke/iter_20.pth
```

**验收**：路径均存在；`python -c "import mmseg; import torch; print(torch.cuda.is_available())"` 为 `True`。

---

### 步骤 3.1：编写完整配置 `segman_b_trans10k_lass.py`

**做什么**：合并阶段 1 的 LASS backbone 与阶段 2 的 MMSCopE decode head，继承 `segman_b_trans10k.py` 的训练超参（lr、80k、Trans10K 数据）。

**目的**：提供阶段 3 **唯一正式配置**，与冒烟用的 `*_enc_only` / `*_dec_only` 区分。

**涉及文件**：`local_configs/segman_trans/segman_b_trans10k_lass.py`（新建）

**推荐内容**：

```python
_base_ = ['./segman_b_trans10k.py']

model = dict(
    type='EncoderDecoderLASS',
    backbone=dict(
        type='SegMANEncoderLASS_b',
        pretrained='/workspace/segman/pretrained/SegMAN_Encoder_b.pth.tar',
        style='pytorch',
        lass_cfg=dict(
            enable_stages=[0, 1, 2],
            enable_ltab=True,
            enable_rsm=True,
            dilate_kernel=5,
            ltab=dict(beta_init=0.1, alpha_init=1.0, tau_init=0.0),
            rsm=dict(gamma_init=0.5, delta_init=0.5),
        ),
    ),
    decode_head=dict(
        type='SegMANDecoderMMSCopE',
        mmscope_cfg=dict(
            enable_bpm=True,
            enable_msbec=True,
            boundary_loss_weight=0.4,
            refine_eta=0.1,
            dilate_kernel=5,
            erode_kernel=5,
        ),
    ),
)

# 与基线 B4 一致
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=8000, metric='mIoU', save_best='mIoU')
data = dict(samples_per_gpu=2, workers_per_gpu=4)
```

**命令**：

```bash
python tools/print_config.py local_configs/segman_trans/segman_b_trans10k_lass.py | head -80
```

**验收标准**：

- `model.type` = `EncoderDecoderLASS`
- `backbone.type` = `SegMANEncoderLASS_b`
- `decode_head.type` = `SegMANDecoderMMSCopE`
- `runner.max_iters` = 80000，`num_classes` = 12

---

### 步骤 3.2（可选）：拆分 B7-1 / B7-2 子配置

**做什么**：复制 `segman_b_trans10k_lass.py` 为：

- `segman_b_trans10k_lass_b7-1.py`：`max_iters=40000`，可选在 `optimizer.paramwise_cfg` 中对 **Stage4**（`layers.6` 全局注意力块）设 `lr_mult=0` 或 `decay_mult=0` 近似冻结
- `segman_b_trans10k_lass_b7-2.py`：`max_iters=80000`，全网正常 lr（用于从 B7-1 的 `iter_40000.pth` resume）

**目的**：落实设计书 §8.2 的 S0/S1/S2——先让 **LASS + BPM** 在较稳定 backbone 上适应，再端到端精调。

**说明**：若暂不实现 Stage4 冻结，B7-1 仍可用 **较低全局 lr** 代替，例如 `--cfg-options optimizer.lr=3e-5`。

---

### 步骤 3.3：方案 A — 一键正式训练 80k（推荐）

**做什么**：从基线 80k 权重热启动，训练完整路线 B 模型至 80000 iter。

**目的**：得到可与基线直接对比的 **`iter_80000.pth`**，验证 mIoU 是否超过 80.71%。

**命令（前台）**：

```bash
cd /workspace/segman/segmentation
conda activate segman

python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_lass.py \
  --work-dir outputs/trans10k_lass_mmscope \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate \
  --cfg-options data.workers_per_gpu=2
```

**命令（后台 nohup，与 B4 相同）**：

```bash
nohup python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_lass.py \
  --work-dir outputs/trans10k_lass_mmscope \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate \
  --cfg-options data.workers_per_gpu=2 \
  > outputs/trans10k_lass_mmscope/train.log 2>&1 &

echo $! > outputs/trans10k_lass_mmscope/train.pid
tail -f outputs/trans10k_lass_mmscope/train.log
```

**续训**（中断后）：

```bash
python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_lass.py \
  --work-dir outputs/trans10k_lass_mmscope \
  --resume-from outputs/trans10k_lass_mmscope/iter_40000.pth \
  --no-validate \
  --cfg-options data.workers_per_gpu=2
```

**Docker 资源（与 B4 相同）**

- 启动容器：`--shm-size=8g --memory=32g`
- 训练时 **`--no-validate`**：避免验证阶段 OOM `Killed`（与基线 80k 经验一致）
- 每 **4000 iter** 存盘（`checkpoint_config.interval=4000`）

**训练过程看什么**

| 日志项 | 含义 |
|--------|------|
| `decode.loss_ce` 或 `loss_seg` | 语义分割 CE |
| `decode.loss_bd` | 边界损失（MMSCopE） |
| `Iter [k/80000]` | 进度；单卡约需数小时～一天量级（视 GPU 而定） |

**验收标准（训练结束）**：

- 生成 `outputs/trans10k_lass_mmscope/iter_80000.pth`
- 日志无 `Killed`、无持续 NaN

**说明**：`load-from` 时打印的 `missing keys`（LTAB/RSM/BPM/MSBEC）与 `unexpected keys`（classifier）为 **正常现象**。

---

### 步骤 3.4：方案 B — B7-1（0～40k，偏稳定起步）

**做什么**：第一阶段：在加载基线权重前提下，优先适应 **LASS + 解码器新模块**；建议 **冻结或弱化 Stage4**（全局注意力，预训练最敏感）。

**目的**：减少早期破坏 ImageNet/基线已学好的高层全局表征，与设计书 S0/S1 一致。

**命令**：

```bash
python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_lass.py \
  --work-dir outputs/trans10k_lass_mmscope_b7-1 \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate \
  --cfg-options runner.max_iters=40000 data.workers_per_gpu=2 optimizer.lr=6e-5
```

若已建 `segman_b_trans10k_lass_b7-1.py`，将上面 config 路径换成该文件。

**验收**：`outputs/trans10k_lass_mmscope_b7-1/iter_40000.pth` 存在。

---

### 步骤 3.5：方案 B — B7-2（40k～80k，全网精调）

**做什么**：从 B7-1 的 checkpoint **续训** 40000 iter；学习率建议降为原来的 **0.1×**（设计书 §8.2 S2）。

**目的**：端到端微调 LASS + MMSCopE + 原 SegMAN 共有层，冲击更高 mIoU。

**命令**：

```bash
python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_lass.py \
  --work-dir outputs/trans10k_lass_mmscope \
  --resume-from outputs/trans10k_lass_mmscope_b7-1/iter_40000.pth \
  --no-validate \
  --cfg-options runner.max_iters=80000 data.workers_per_gpu=2 optimizer.lr=6e-6
```

**说明**：`resume-from` 会继承优化器状态；若希望 **重新计数 iter 0～40k** 仅加载权重，可改用 `--load-from iter_40000.pth` 并设 `runner.max_iters=40000`，再手动链第二次训练。团队内选一种约定即可。

**验收**：`outputs/trans10k_lass_mmscope/iter_80000.pth`（或当前 work-dir 下最终 iter）。

---

### 步骤 3.6：方案 B — B7-3（进阶，可选）

**做什么**：在设计书建议下将 RSM 的 `M_bg` 从 **仅 GT** 扩展为 **`gt + pred` 融合**（需改 `lass_cfg` / `EncoderDecoderLASS`）；并可加强边界相关数据增强。

**目的**：后期让反射抑制更贴近推理分布（无 GT 时仍可用预测 mask 估计背景）。

**当前仓库状态**：阶段 1～2 默认 **训练全程 `M_bg` 来自 GT**；本步为 **增强项**，未实现前可跳过，不影响方案 A 完整 80k。

**若已实现 `bg_mask_mode`**，示例：

```python
lass_cfg=dict(..., rsm=dict(..., bg_mask_mode='gt+pred'))
```

**验收**：对比 B7-2 与 B7-3 的 val mIoU 与 window/shelf/box 分项（见步骤 3.7）。

---

### 步骤 3.7：验证集评测（必做）

**做什么**：训练若使用了 `--no-validate`，必须在训完后单独跑 `tools/test.py`。

**目的**：得到与基线可比的 **mIoU / 每类 IoU**，判断是否超过 80.71% 及弱类是否提升。

**正式权重（推荐）**：

```bash
cd /workspace/segman/segmentation
conda activate segman

python tools/test.py \
  local_configs/segman_trans/segman_b_trans10k_lass.py \
  --checkpoint outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --eval mIoU
```

**注意**：checkpoint 必须用 **`--checkpoint`**，不要写成位置参数（与 B4 相同）。**勿用** `trans10k_lass_mmscope/iter_80000.pth`（无 ignore_index 修复，test mIoU ≈ 16%）。

**已记录结果（fix5k @ iter_5000，Trans10K val 1000 张）**：

- **基线 vs 5k 全量对比**（总体 / 12 类 IoU / 12 类 Acc / 升降统计）：见上文 **§0.1**。
- **验收摘要**：mIoU **80.84%**（+0.13）☑；window **76.27%**（+9.65）☑；shelf、box 与基线持平（±0.2 以内）✗ 未超基线。

记录结果可同步写入 `Trans10K_SegMAN_B_训练与评测结果.md` 或实验表格。

---

### 步骤 3.8：可视化与交付（可选）

**做什么**：对典型透明场景（window / shelf / box）对比基线与路线 B 的 mask、边界图 `P_bd`。

**目的**：论文/报告图示；确认 MMSCopE 边界更清晰而不仅是 mIoU 数字提升。

```bash
python tools/test.py \
  local_configs/segman_trans/segman_b_trans10k_lass.py \
  --checkpoint outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --eval mIoU \
  --show-dir outputs/trans10k_lass_mmscope_fix5k/vis_test
```

---

### 步骤 3.9：进入路线 C 前的检查

**做什么**：确认最佳分割权重路径、配置名、输入分辨率与下游 `transgrasp` 推理接口一致。

**目的**：路线 C（Grounded-SAM + TransFine）**只消费 mask/ROI**，需固定「用哪一版 checkpoint」。

**交付物（项目后续统一以 fix5k 为准，见《路线B_fix5k_项目后续步骤.md》）**：

- **正式部署权重**：`outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth`（mIoU **80.84%**，bowl **80.07%**，§0.1）
- 配置：`segman_b_trans10k_lass.py`
- **归档对照（不部署）**：`balanced10k/iter_10000.pth`（mIoU 81.76%，bowl 74.31%，§0.3）
- 推理：输出 **12 类语义图**；下游可取 **非 background 并集** 或 dominant class 作为透明 mask

---

### 阶段 3 完成检查表

| 步骤 | 状态 | 验收 |
|------|------|------|
| 3.0 环境与前置 | ☑ | 数据 + `iter_80000` + 阶段 2 冒烟 ckpt |
| 3.1 `segman_b_trans10k_lass.py` | ☑ | `print_config` 类型正确 |
| ignore_index=255 修复 | ☑ | `lass_utils` / `mmscope` / 调用链 |
| **5k 微调** | ☑ | `fix5k/iter_5000.pth`，mIoU 80.84% |
| **balanced10k** | ☑ | `balanced10k/iter_10000.pth`，mIoU **81.76%**（§0.3） |
| 3.3 方案 A 80k（对照） | ☑ 已跑 | 未达标，见 §0.2；**勿部署** |
| 3.4～3.5 方案 B（可选） | ☐ | B7-1/2 链式 ckpt |
| 3.7 `test.py` mIoU | ☑ | fix5k 80.84%；balanced10k **81.76%** |
| 3.9 路线 C 交接 | ☑ | **正式 fix5k**；后续见《路线B_fix5k_项目后续步骤.md》 |

**阶段 3 通过后** → 路线 **C**（`transgrasp`、Grounded-SAM、TransFine），见《项目实施步骤指南.md》。

---

### 阶段 3 与基线 / 阶段 1～2 对照

| 项目 | 基线 B4 | 阶段 3 路线 B |
|------|---------|----------------|
| Config | `segman_b_trans10k.py` | `segman_b_trans10k_lass.py` |
| Segmentor | `EncoderDecoder` | `EncoderDecoderLASS` |
| Backbone | `SegMANEncoder_b` | `SegMANEncoderLASS_b` |
| Decode head | `SegMANDecoder` | `SegMANDecoderMMSCopE` |
| 损失 | CE | CE + `loss_bd` |
| work-dir | `outputs/trans10k_segman_b` | `outputs/trans10k_lass_mmscope` |
| 热启动 | ImageNet pretrained | pretrained + **`--load-from iter_80000.pth`** |

---

### 阶段 3：完整训练（B7）— 约 1 周（总览）

| 步骤 | 任务 | 产出 |
|------|------|------|
| 3.0 | 环境 + 前置检查 | — |
| 3.1 | 完整 config | `segman_b_trans10k_lass.py` |
| 3.2 | B7 子 config（可选） | `*_b7-1.py` 等 |
| 3.3 | **方案 A** 80k | `trans10k_lass_mmscope/iter_80000.pth` |
| 3.4～3.6 | **方案 B** 分阶段（可选） | B7-1/2/3 ckpt |
| 3.7 | `test.py` | mIoU 表、类 IoU |
| 3.8 | 可视化（可选） | `vis_val/` |
| 3.9 | 路线 C 交接 | 文档记录 best ckpt |

---

## 3. 文件清单（待创建）

```text
segmentation/mmseg/models/modules/
├── __init__.py
├── ltab.py                 # 阶段 1.1（已建骨架）
├── rsm.py                  # 阶段 1.2（已建骨架）
└── mmscope.py              # 阶段 2.1～2.2（已实现）

segmentation/mmseg/models/backbones/
└── segman_encoder_lass.py  # 阶段 1.3～1.4

segmentation/mmseg/models/decode_heads/
└── segman_decoder_mmscope.py  # 阶段 2.3（已实现）

segmentation/local_configs/segman_trans/
├── segman_b_trans10k_lass_enc_only.py      # 阶段 1（已有）
├── segman_b_trans10k_mmscope_dec_only.py       # 阶段 2.6a：基线 backbone + MMSCopE（优先）
├── segman_b_trans10k_lass_mmscope_dec_only.py  # 阶段 2.6b：LASS + MMSCopE 联合冒烟
├── segman_b_trans10k_lass.py                   # 阶段 3.1：完整路线 B 80k（已建）
├── segman_b_trans10k_lass_b7-1.py              # 阶段 3.2 可选：B7-1 40k
└── segman_b_trans10k_lass_b7-2.py              # 阶段 3.2 可选：B7-2 续训
```

---

## 4. 阶段 1 命令速查（复制块）

```bash
docker exec -it segman_train bash
conda activate segman
cd /workspace/segman/segmentation

# 1.1 + 1.2
python -c "from mmseg.models.modules.ltab import LTAB; from mmseg.models.modules.rsm import ReflectionSuppression; import torch; x=torch.randn(2,64,64,64); assert LTAB(64)(x).shape==x.shape; m=torch.zeros(2,1,64,64); m[:,:,:20,:]=1; assert ReflectionSuppression(64)(x,m).shape==x.shape; print('1.1-1.2 ok')"

# 1.4（实现并注册 SegMANEncoderLASS_b 后）
python -c "from mmseg.models import build_backbone; import torch; m=build_backbone(dict(type='SegMANEncoderLASS_b',pretrained=None,style='pytorch')); m.eval(); print([t.shape for t in m(torch.randn(1,3,512,512))])"

# 1.5（实现 lass_enc_only 配置后）
python tools/train.py local_configs/segman_trans/segman_b_trans10k_lass_enc_only.py --work-dir outputs/trans10k_lass_enc_smoke --cfg-options runner.max_iters=20 data.workers_per_gpu=2 --load-from outputs/trans10k_segman_b/iter_80000.pth
```

---

## 4b. 阶段 2 命令速查（复制块）

```bash
docker exec -it segman_train bash
source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman
cd /workspace/segman/segmentation

# 2.1 BPM
python -c "from mmseg.models.modules.mmscope import BoundaryProbabilityModule; import torch; m=BoundaryProbabilityModule(180); x=torch.randn(2,180,64,64); p,w=m(x); assert p.shape==(2,1,64,64); print('2.1 ok')"

# 2.2 MSBEC（GPU）
python -c "from mmseg.models.modules.mmscope import MultiScaleBoundaryEnhance; import torch; m=MultiScaleBoundaryEnhance(180).cuda(); f=torch.randn(2,180,64,64,device='cuda',requires_grad=True); p=torch.rand(2,1,64,64,device='cuda'); o=m(f,p); o.sum().backward(); print('2.2 ok')"

# 2.4 Y_bd
python -c "from mmseg.models.modules.mmscope import semantic_seg_to_boundary; import torch; y=torch.zeros(2,1,64,64,dtype=torch.long); y[:,:,20:40,20:40]=3; bd=semantic_seg_to_boundary(y); print('2.4 ok', bd.sum().item())"

# 2.6a 冒烟（基线 backbone + MMSCopE，必做 · 优先）
python tools/train.py local_configs/segman_trans/segman_b_trans10k_mmscope_dec_only.py \
  --work-dir outputs/trans10k_mmscope_dec_smoke \
  --cfg-options runner.max_iters=20 data.workers_per_gpu=2 \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate

# 2.6b 冒烟（LASS + MMSCopE，2.6a 通过后再做）
python tools/train.py local_configs/segman_trans/segman_b_trans10k_lass_mmscope_dec_only.py \
  --work-dir outputs/trans10k_lass_mmscope_dec_smoke \
  --cfg-options runner.max_iters=20 data.workers_per_gpu=2 \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate
```

---

## 4c. 阶段 3 命令速查（复制块）

```bash
docker exec -it segman_train bash
source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman
cd /workspace/segman/segmentation

# 3.1 检查配置（需先创建 segman_b_trans10k_lass.py）
python tools/print_config.py local_configs/segman_trans/segman_b_trans10k_lass.py

# 3.3 方案 A：正式 80k（推荐）
nohup python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_lass.py \
  --work-dir outputs/trans10k_lass_mmscope \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate \
  --cfg-options data.workers_per_gpu=2 \
  > outputs/trans10k_lass_mmscope/train.log 2>&1 &

# 3.7 评测（正式权重 fix5k）
python tools/test.py \
  local_configs/segman_trans/segman_b_trans10k_lass.py \
  --checkpoint outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --eval mIoU

# 5k 微调（推荐交付）
python tools/train.py local_configs/segman_trans/segman_b_trans10k_lass.py \
  --work-dir outputs/trans10k_lass_mmscope_fix5k \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate \
  --cfg-options runner.max_iters=5000 data.workers_per_gpu=2 optimizer.lr=3e-5
```

---

## 5. 与路线 C 的关系

路线 B **只改分割网络**；mIoU 达标后，路线 C 仍用 **最佳分割 checkpoint**（**`outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth`**）出 mask → Grounded-SAM → TransFine，逻辑不变。

---

## 6. 当前进度勾选

- 基线 SegMAN-B 80k 训练与评测（mIoU 80.71%）
- 1.1 LTAB 模块（`ltab.py`）
- 1.2 RSM 模块（`rsm.py`）
- 1.3～1.4 SegMANEncoderLASS_b
- 1.5 backbone-only 冒烟（`outputs/trans10k_lass_enc_smoke`）
- 2.0～2.4 MMSCopE 模块与边界 GT（`mmscope.py`）
- 2.3 SegMANDecoderMMSCopE 注册
- 2.5 loss_bd + 双 config（`mmscope_dec_only` / `lass_mmscope_dec_only`）
- **2.6a** 基线 backbone + MMSCopE 冒烟（`trans10k_mmscope_dec_smoke`）
- **2.6b** LASS + MMSCopE 联合冒烟（`trans10k_lass_mmscope_dec_smoke`）
- 2.7 消融（可选）
- 3.0 环境与前置检查
- 3.1 `segman_b_trans10k_lass.py`（已建）
- ignore_index=255 修复 + `verify_ignore_index_fix.py`
- **5k 微调 + test**：`fix5k/iter_5000.pth`，mIoU **80.84%**（§0.1）
- **balanced10k + test**：`balanced10k/iter_10000.pth`，mIoU **81.76%**（§0.3）
- 3.3 方案 A 80k（对照实验，未达标，§0.2）
- 3.4～3.6 方案 B 分阶段（可选）
- 3.7 `test.py` mIoU > 80.71%（☑，最高 81.76%）
- 3.9 路线 C 交接（主推 `balanced10k/iter_10000.pth`）

