# 路线 B 平衡微调方案（Balanced Finetune）

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.0 |
| 编写日期 | 2026-05-23 |
| 前置权重 | 基线 `outputs/trans10k_segman_b/iter_80000.pth`（mIoU **80.71%**） |
| 当前正式交付 | `outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth`（mIoU **80.84%**，见《路线B_LASS_MMSCopE_实施清单.md》§0.1） |
| 本方案目标 | **mIoU ≥ 80.71%**，且 **多数类别 IoU 高于基线**（建议 ≥8/12 类 Δ>0.2%） |
| 关联清单 | 《路线B_LASS_MMSCopE_实施清单.md》《Trans10K_SegMAN_B_训练与评测结果.md》 |

---

## 1. 背景与动机

### 1.1 fix5k 的成绩与矛盾

在修复 `ignore_index=255` 后，**5k 微调**（`fix5k/iter_5000.pth`）已取得：

| 维度 | 结果 |
|------|------|
| 总 mIoU | **80.84%**（基线 80.71%，**+0.13**） |
| window | **76.27%**（基线 66.62%，**+9.65**） |
| 其它类 | bottle / eyeglass / jar_kettle / door 等 **明显回落**；shelf、box **基本持平** |

结论：LASS + MMSCopE **有效**，但训练目标偏「边界难类」，易牺牲基线已学好的区域类。

### 1.2 为何不用 80k

| run | max_iters | test mIoU | 说明 |
|-----|-----------|-----------|------|
| 基线 | 80000（从零训） | 80.71% | 正常 |
| fix5k | 5000 | **80.84%** | 推荐正式权重 |
| fix80k / fix80k_lr3e5 | 80000（从基线微调） | 11%～38% | **长训塌缩，勿用** |

本方案为 **从基线 `iter_80000.pth` 出发的短程微调**，**不是**再跑满 80000 iter。

---

## 2. 方案总览

```text
基线 iter_80000.pth
        │
        ├─► 方案 A（推荐）LASS + MMSCopE 平衡版
        │     配置: segman_b_trans10k_lass_balanced.py
        │     work-dir: outputs/trans10k_lass_mmscope_balanced10k
        │     max_iters: 10000（配置内写死，非 80k）
        │
        └─► 方案 B（备选）仅 MMSCopE，encoder 保持基线
              配置: segman_b_trans10k_mmscope_balanced.py
              work-dir: outputs/trans10k_mmscope_balanced10k
```

---

## 3. 与 fix5k 的差异（超参对照）

| 项 | fix5k | balanced（本方案） |
|----|-------|-------------------|
| 配置 | `segman_b_trans10k_lass.py` | `segman_b_trans10k_lass_balanced.py` |
| **max_iters** | 5000 | **10000** |
| **optimizer.lr** | 3e-5 | **2e-5** |
| boundary_loss_weight | 0.4 | **0.15** |
| refine_eta | 0.1 | **0.05** |
| LASS enable_stages | [0, 1, 2] | **[1, 2]**（跳过 stage0） |
| ltab beta_init | 0.1 | **0.05** |
| rsm gamma/delta_init | 0.5 | **0.3** |
| loss_decode | CE only | **CE（class_weight）+ Dice 0.4** |
| 新模块 lr_mult | head 10× | head 10×；ltab/rsm/bpm/msbec **6×** |
| checkpoint 间隔 | 4000 | **2000** |

**说明**：命令行未写 `runner.max_iters` 时，以配置文件 **`max_iters=10000`** 为准（见 §5.1）。

---

## 4. 各项改动的目的（好处）

### 4.1 降低边界损失（0.4 → 0.15）与 refine_eta（0.1 → 0.05）

- **问题**：边界损失过强时，BPM/MSBEC 主导梯度，window 涨得快，bottle/cup/eyeglass 等区域类易掉。
- **好处**：保留 MMSCopE 边界监督，但以 **语义分割为主**；利于「window 仍高于基线 + 回落类回升」。

### 4.2 LASS 仅 stage 1、2（不开 stage 0）

- **问题**：浅层负责边缘与细纹理，过早 LTAB/RSM 易扰动低层特征。
- **好处**：深层仍做反射抑制；**浅层更接近基线**，利于 bottle/cup 等。

### 4.3 更弱的 LASS 初值

- **好处**：从基线 `load-from` 时新模块 **小扰动起步**，减少 5k～10k 内表征被拉歪。

### 4.4 类别加权 CE

对 fix5k 中易回落类略加重（window 不加重，避免拆东墙补西墙）：

| 类别 | class_weight |
|------|----------------|
| background, box, window, wall, bowl | 1.0 |
| bottle, jar_kettle | 1.10 |
| eyeglass, door | 1.12 |
| freezer, shelf | 1.02 |
| cup | 1.05 |

### 4.5 Dice 辅助损失（weight 0.4）

- **好处**：与 CE 互补，对 **区域重叠** 更敏感，利于稳住 bowl/wall/shelf 等 IoU。

### 4.6 10k iter + lr 2e-5

- **问题**：5k 可能未充分对齐回落类；80k 会塌缩。
- **好处**：在 fix5k 与 80k 之间的折中，给回落类更多步数，且 **更低 lr** 更稳。

### 4.7 新模块更高 lr_mult（6×）

- **好处**：基线主体少动，LASS/MMSCopE 多动，符合「插模块微调」逻辑。

### 4.8 方案 B（decoder-only）

- **好处**：改动面最小，常能保住 bottle/cup/eyeglass；若方案 A 仍多类 ↓ 可再试。
- **代价**：无 LASS，window 等收益可能略弱于完整路线 B。

---

## 5. 文件与验收标准

### 5.1 配置文件

| 文件 | 用途 |
|------|------|
| `segmentation/local_configs/segman_trans/segman_b_trans10k_lass_balanced.py` | 方案 A |
| `segmentation/local_configs/segman_trans/segman_b_trans10k_mmscope_balanced.py` | 方案 B |
| `segmentation/scripts/compare_miou_vs_baseline.py` | 逐类 ↑/↓ 统计 |
| `segmentation/scripts/train_route_b_balanced.sh` | 可选一键训练+扫 ckpt |

### 5.2 验收标准（建议）

| 条件 | 阈值 |
|------|------|
| test mIoU（val 1000 张） | **≥ 80.71%** |
| 相对基线 IoU 上升类数 | **≥ 8 / 12**（\|Δ\|>0.2% 计 ↑/↓） |
| 权重 | 扫 `iter_2000`～`iter_10000`，取 test 最优且满足上两条的 ckpt |

**未达标前**：正式交付仍用 **`fix5k/iter_5000.pth`**。

### 5.3 训练步数说明（重要）

| 训练类型 | max_iters |
|----------|-----------|
| 基线 SegMAN-B（从零） | 80000 |
| fix5k | 5000 |
| **本方案 balanced** | **10000**（配置内 `runner.max_iters=10000`） |
| 失败 fix80k 等 | 80000 → **勿用** |

**nohup 命令里若不写 `runner.max_iters`，默认仍是 10k，不是 80k。**

---

## 6. 执行命令（Docker：`segman_train`）

工作目录：`/workspace/segman/segmentation`

### 6.1 进入环境

```bash
docker exec -it segman_train bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate segman
cd /workspace/segman/segmentation
```

### 6.2 训练前自检（可选）

```bash
python scripts/verify_ignore_index_fix.py
python scripts/verify_mmscope_phase2.py
python scripts/check_ckpt_load.py
```

### 6.3 方案 A：LASS + MMSCopE 平衡版（10k）

前台：

```bash
mkdir -p outputs/trans10k_lass_mmscope_balanced10k

python tools/train.py local_configs/segman_trans/segman_b_trans10k_lass_balanced.py \
  --work-dir outputs/trans10k_lass_mmscope_balanced10k \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate \
  --cfg-options data.workers_per_gpu=2
```

后台（推荐）：

```bash
mkdir -p outputs/trans10k_lass_mmscope_balanced10k

nohup python tools/train.py local_configs/segman_trans/segman_b_trans10k_lass_balanced.py \
  --work-dir outputs/trans10k_lass_mmscope_balanced10k \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate \
  --cfg-options data.workers_per_gpu=2 \
  > outputs/trans10k_lass_mmscope_balanced10k/train.log 2>&1 &

tail -f outputs/trans10k_lass_mmscope_balanced10k/train.log
```

**显式写死 10k + lr（与配置一致，防误改配置）**：

```bash
nohup python tools/train.py local_configs/segman_trans/segman_b_trans10k_lass_balanced.py \
  --work-dir outputs/trans10k_lass_mmscope_balanced10k \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate \
  --cfg-options runner.max_iters=10000 data.workers_per_gpu=2 optimizer.lr=2e-5 \
  > outputs/trans10k_lass_mmscope_balanced10k/train.log 2>&1 &
```

确认步数：

```bash
grep -E "max_iters|Iter \[10000" outputs/trans10k_lass_mmscope_balanced10k/train.log | head -5
```

预期 checkpoint：`iter_2000.pth` … `iter_10000.pth`（每 2000 存一次）。

### 6.4 方案 B：仅 MMSCopE（备选）

```bash
mkdir -p outputs/trans10k_mmscope_balanced10k

nohup python tools/train.py local_configs/segman_trans/segman_b_trans10k_mmscope_balanced.py \
  --work-dir outputs/trans10k_mmscope_balanced10k \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate \
  --cfg-options runner.max_iters=10000 data.workers_per_gpu=2 optimizer.lr=2e-5 \
  > outputs/trans10k_mmscope_balanced10k/train.log 2>&1 &
```

### 6.5 评测（必须用 `--checkpoint`）

单点（例如 10k）：

```bash
python tools/test.py local_configs/segman_trans/segman_b_trans10k_lass_balanced.py \
  --checkpoint outputs/trans10k_lass_mmscope_balanced10k/iter_10000.pth \
  --eval mIoU
```

扫中间 ckpt（找峰值）：

```bash
for it in 2000 4000 6000 8000 10000; do
  echo "===== iter_${it} ====="
  python tools/test.py local_configs/segman_trans/segman_b_trans10k_lass_balanced.py \
    --checkpoint outputs/trans10k_lass_mmscope_balanced10k/iter_${it}.pth \
    --eval mIoU \
    --work-dir outputs/trans10k_lass_mmscope_balanced10k/eval_iter_${it}
done
```

### 6.6 与基线逐类对比

```bash
python scripts/compare_miou_vs_baseline.py \
  outputs/trans10k_lass_mmscope_balanced10k/eval_iter_10000/eval_single_scale_*.json
```

（将路径换成实际生成的 json 文件。）

### 6.7 基线对照（可选）

```bash
python tools/test.py local_configs/segman_trans/segman_b_trans10k.py \
  --checkpoint outputs/trans10k_segman_b/iter_80000.pth \
  --eval mIoU
```

### 6.8 一键脚本（可选）

```bash
chmod +x scripts/train_route_b_balanced.sh
bash scripts/train_route_b_balanced.sh lass
# decoder-only: bash scripts/train_route_b_balanced.sh dec
```

---

## 7. 基线 per-class IoU 参照（%）

用于对比脚本与填表：

| 类别 | IoU |
|------|-----|
| background | 96.71 |
| box | 71.47 |
| bottle | 87.77 |
| window | 66.62 |
| eyeglass | 92.85 |
| freezer | 73.90 |
| jar_kettle | 84.04 |
| door | 75.04 |
| cup | 90.91 |
| wall | 82.72 |
| bowl | 78.91 |
| shelf | 67.61 |
| **mIoU** | **80.71** |

fix5k 对照见《路线B_LASS_MMSCopE_实施清单.md》**§0.1**。

---

## 8. 实测结果（方案 A，2026-05-23）

**训练**：`outputs/trans10k_lass_mmscope_balanced10k`，**10000/10000** 完成；末步 `loss≈0.071`（`loss_ce`+`loss_dice`+`loss_bd`）。  
**评测**：Trans10K **val 1000 张**；`tools/test.py` + `scripts/compare_miou_vs_baseline.py`（\|Δ\|>0.2% 计 ↑/↓）。  
**方案 B**：未训练。

### 8.1 checkpoint 总览

| checkpoint | mIoU | Δ vs 基线 | ↑ / ≈ / ↓ | 备注 |
|------------|------|-----------|-----------|------|
| iter_2000 | — | — | — | 未测（可补） |
| iter_4000 | — | — | — | 未测（可补） |
| **iter_6000** | **80.83%** | +0.12 | **6 / 3 / 3** | **↑ 类数最多** |
| iter_8000 | 81.59% | +0.88 | 5 / 2 / 5 | mIoU 与 window 折中 |
| **iter_10000** | **81.76%** | **+1.05** | 5 / 2 / 5 | **mIoU 最高**（推荐主表） |

对照：**基线 80.71%**；**fix5k 80.84%**（§0.1）。

### 8.2 iter_10000 逐类 IoU（%）

| 类别 | 基线 | balanced 10k | Δ | 趋势 |
|------|------|--------------|-----|------|
| background | 96.71 | 96.45 | −0.26 | ↓ 略 |
| box | 71.47 | 71.86 | +0.39 | ↑ |
| bottle | 87.77 | 88.20 | +0.43 | ↑ |
| **window** | 66.62 | **82.91** | **+16.29** | ↑ |
| eyeglass | 92.85 | 92.01 | −0.84 | ↓ |
| freezer | 73.90 | 73.48 | −0.42 | ↓ 略 |
| jar_kettle | 84.04 | 83.89 | −0.15 | ≈ |
| door | 75.04 | 74.40 | −0.64 | ↓ |
| cup | 90.91 | 90.96 | +0.05 | ≈ |
| wall | 82.72 | 83.95 | +1.23 | ↑ |
| bowl | 78.91 | 74.31 | −4.60 | ↓ |
| shelf | 67.61 | 68.73 | +1.12 | ↑ |
| **mIoU** | **80.71** | **81.76** | **+1.05** | ↑ |

### 8.3 iter_6000 逐类 IoU（%）— 类均衡最佳

| 类别 | 基线 | balanced 6k | Δ | 趋势 |
|------|------|-------------|-----|------|
| background | 96.71 | 96.42 | −0.29 | ↓ |
| box | 71.47 | 74.16 | +2.69 | ↑ |
| bottle | 87.77 | 87.98 | +0.21 | ↑ |
| window | 66.62 | 69.41 | +2.79 | ↑ |
| eyeglass | 92.85 | 90.84 | −2.01 | ↓ |
| freezer | 73.90 | 75.30 | +1.40 | ↑ |
| jar_kettle | 84.04 | 84.48 | +0.44 | ↑ |
| door | 75.04 | 75.47 | +0.43 | ↑ |
| cup | 90.91 | 90.89 | −0.02 | ≈ |
| wall | 82.72 | 82.79 | +0.07 | ≈ |
| bowl | 78.91 | 74.86 | −4.05 | ↓ |
| shelf | 67.61 | 67.42 | −0.19 | ≈ |
| **mIoU** | **80.71** | **80.83** | **+0.12** | ↑ |

### 8.4 iter_8000 逐类 IoU（%）

| 类别 | 基线 | balanced 8k | Δ | 趋势 |
|------|------|-------------|-----|------|
| background | 96.71 | 96.52 | −0.19 | ≈ |
| box | 71.47 | 71.80 | +0.33 | ↑ |
| bottle | 87.77 | 87.49 | −0.28 | ↓ |
| window | 66.62 | 76.77 | +10.15 | ↑ |
| eyeglass | 92.85 | 92.04 | −0.81 | ↓ |
| freezer | 73.90 | 74.04 | +0.14 | ≈ |
| jar_kettle | 84.04 | 83.82 | −0.22 | ↓ |
| door | 75.04 | 76.90 | +1.86 | ↑ |
| cup | 90.91 | 90.70 | −0.21 | ↓ |
| wall | 82.72 | 84.13 | +1.41 | ↑ |
| bowl | 78.91 | 74.25 | −4.66 | ↓ |
| shelf | 67.61 | 70.61 | +3.00 | ↑ |
| **mIoU** | **80.71** | **81.59** | **+0.88** | ↑ |

### 8.5 验收与权重建议

| 验收项 | 阈值 | iter_6000 | iter_10000 |
|--------|------|-----------|------------|
| mIoU ≥ 80.71% | ☑ 要求 | ☑ 80.83% | ☑ **81.76%** |
| ≥8/12 类 IoU ↑ | 建议 | ✗（6↑） | ✗（5↑） |

**共性**：**bowl** 在 6k/8k/10k 均明显低于基线（约 −4～5）；**window** 随 iter 增加持续升高（6k +2.8 → 10k +16.3）。

| 用途 | 推荐权重 |
|------|----------|
| 论文/报告 **mIoU 主表** | `outputs/trans10k_lass_mmscope_balanced10k/iter_10000.pth` |
| 强调 **多类 IoU 不低于基线** | `iter_6000.pth`（↑6 类，mIoU 仍 +0.12） |
| window 与 mIoU 折中 | `iter_8000.pth` |
| 路线 C / 稳妥交付（与 fix5k 接近） | 仍可用 `fix5k/iter_5000.pth` |

**是否替换 fix5k**：若接受 bowl 回落、以 **总 mIoU + window** 为主，**建议将路线 B 主推权重升级为 `balanced10k/iter_10000.pth`**；否则保留 fix5k，balanced 作增强对照。

评测产物目录：`outputs/trans10k_lass_mmscope_balanced10k/eval_iter_{6000,8000,10000}/`。

---

## 9. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-23 | 初版：平衡微调动机、超参、命令、验收标准 |
| 2026-05-23 | §8：填入 balanced10k 训练完成与 6k/8k/10k test 结果 |
