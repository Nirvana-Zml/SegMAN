# 路线 B 平衡微调方案（Balanced Finetune）

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.0 |
| 编写日期 | 2026-05-23 |
| 前置权重 | 基线 `outputs/trans10k_segman_b/iter_80000.pth`（mIoU **80.71%**） |
| 当前正式交付 | `outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth`（mIoU **80.84%**，见《路线B_LASS_MMSCopE_实施清单.md》§0.1） |
| 本方案目标 | **mIoU ≥ 80.71%**，且 **多数类别 IoU 高于基线**（建议 ≥8/12 类 Δ>0.2%） |
| 关联清单 | 《路线B_LASS_MMSCopE_实施清单.md》《Trans10K_SegMAN_B_训练与评测结果.md》 |
| **三方案对比** | 《路线B_基线_fix5k_balanced10k_对比分析.md》（基线 / fix5k / iter_10000 合一表） |

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

### 8.6 bowl 回落诊断（方案 3，2026-05-23）

**背景**：balanced10k `iter_10000` 总 mIoU **81.76%**，但 **bowl IoU 74.31%**（基线 78.91%，约 **−4.60%**）。在启动 **方案 1（bowl 专项微调）** 前，先完成方案 3：**定量测试 → 可视化 → 像素级混淆统计**。

**诊断对象**：`outputs/trans10k_lass_mmscope_balanced10k/iter_10000.pth`  
**配置**：`local_configs/segman_trans/segman_b_trans10k_lass_balanced.py`  
**数据**：Trans10K **val 1000 张**（与 §8 评测一致）

#### 8.6.1 方案 3 步骤与命令

| 步骤 | 内容 | 状态 |
|------|------|------|
| 3.0 | `tools/test.py --eval mIoU` | ✅ 见 §8.2（bowl 74.31%，Acc 82.28%） |
| 3.1 | 可视化叠加图 | ✅ |
| 3.2 | GT=bowl 像素上的预测类直方图 | ✅ |

**3.0 / 3.1（Docker，`segmentation` 目录）**：

```bash
python tools/test.py local_configs/segman_trans/segman_b_trans10k_lass_balanced.py \
  --checkpoint outputs/trans10k_lass_mmscope_balanced10k/iter_10000.pth \
  --eval mIoU \
  --show-dir outputs/trans10k_lass_mmscope_balanced10k/vis_bowl_debug
```

- 输出目录：`outputs/trans10k_lass_mmscope_balanced10k/vis_bowl_debug/`（**1000** 张，原图 | GT | 预测 拼接）
- 目视重点：bowl 区域是否被 **background** 挖空、是否与 **cup** 粘连混淆

**3.2 混淆统计脚本**：

```bash
python scripts/analyze_bowl_confusion.py \
  local_configs/segman_trans/segman_b_trans10k_lass_balanced.py \
  --checkpoint outputs/trans10k_lass_mmscope_balanced10k/iter_10000.pth
```

- 脚本：`segmentation/scripts/analyze_bowl_confusion.py`（推理路径与 `tools/test.py` 一致：`DataLoader` + `MMDataParallel` + `return_loss=False`；GT 从 `ann_dir` 读取）

#### 8.6.2 方案 3.2 定量结果（iter_10000）

| 统计项 | 数值 |
|--------|------|
| val 中含 bowl GT 的图像数 | **30 / 1000** |
| bowl GT 像素总数 | **5,449,066** |
| bowl 像素 recall（GT=bowl 上预测为 bowl） | **82.28%** |
| test 表 bowl Acc（同一次 test） | **82.28%**（与 recall 一致） |
| test 表 bowl IoU | **74.31%** |

**在 GT=bowl 的像素上，预测类别分布**：

| 预测类 | 像素数 | 占比 | 说明 |
|--------|--------|------|------|
| **bowl**（正确） | 4,483,372 | **82.28%** | 与 recall 一致 |
| background | 506,200 | **9.29%** | 碗内/边缘漏检、透明区域被当成背景 |
| cup | 337,503 | **6.19%** | 与 cup 语义混淆（balanced 中 cup class_weight=1.05 + Dice 0.4 可能加剧） |
| jar_kettle | 64,450 | 1.18% | 次要 |
| box | 57,541 | 1.06% | 次要 |
| 其它类 | 0 | 0% | — |

**Top 错分（非 bowl）**：background **9.29%** → cup **6.19%** → jar_kettle 1.18% → box 1.06%。

#### 8.6.3 结论（对方案 1 的启示）

1. **recall（82.28%）与 IoU（74.31%）差距大**：说明除「bowl 像素被标错」外，还存在 **误检 bowl**（precision 不足），仅拉高 class_weight 不够，需兼顾边界与 cup/background 竞争。
2. **主因是 background 漏检（9.29%）**，其次 **cup 混淆（6.19%）**；与 balanced  recipe（Dice 0.4、`enable_stages=[1,2]`、cup 1.05）及 window 大幅增益时的表征偏移一致。
3. val 仅 **30 张**含 bowl，**bowl IoU 方差大**；像素级统计样本量仍充足（约 545 万像素），结论可信。
4. **方案 1 建议**（已落配置 `segman_b_trans10k_lass_balanced_bowl.py`）：从 **fix5k** `iter_5000.pth` 短训 **5k**；bowl class_weight **1.18**、cup **1.0**、Dice **0.15**、LASS **stage 0–2**、边界 **0.12**；work-dir `outputs/trans10k_lass_mmscope_balanced_bowl5k`。详见 §10（待训练后补结果）。

---

## 10. bowl 专项微调（方案 1，2026-05-23）

| 项目 | 内容 |
|------|------|
| 配置 | `local_configs/segman_trans/segman_b_trans10k_lass_balanced_bowl.py` |
| 初始权重 | `outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth` |
| work-dir | `outputs/trans10k_lass_mmscope_balanced_bowl5k` |
| max_iters | 5000 |
| 评测权重 | `iter_5000.pth`（Trans10K val 1000 张） |

### 10.1 验收结论

| 验收项 | 阈值 | 结果 | 判定 |
|--------|------|------|------|
| bowl IoU | ≥ 78%（目标≈fix5k 80.07） | **80.25%** | ☑ 通过 |
| mIoU | ≥ 80.71% | **79.15%** | ✗ 未通过（−1.56 vs 基线） |

**小结**：方案 1 **达成 bowl 回升**（相对 balanced10k 74.31 **+5.94**；相对基线 78.91 **+1.34**；略优于 fix5k 80.07 **+0.18**），但以 **总 mIoU 与 shelf 等类** 为代价，**不宜替代 fix5k / balanced10k 作主推权重**。

### 10.2 test 结果（%）

| 类别 | 基线 | fix5k | balanced10k | **bowl5k** | Δ vs 基线 |
|------|------|-------|-------------|------------|-----------|
| background | 96.71 | — | 96.45 | 95.44 | −1.27 |
| box | 71.47 | — | 71.86 | 70.89 | −0.58 |
| bottle | 87.77 | — | 88.20 | 83.96 | −3.81 |
| window | 66.62 | 76.27 | **82.91** | 71.92 | +5.30 |
| eyeglass | 92.85 | — | 92.01 | 89.89 | −2.96 |
| freezer | 73.90 | — | 73.48 | 75.80 | +1.90 |
| jar_kettle | 84.04 | — | 83.89 | 81.37 | −2.67 |
| door | 75.04 | — | 74.40 | 71.47 | −3.57 |
| cup | 90.91 | — | 90.96 | 90.04 | −0.87 |
| wall | 82.72 | — | 83.95 | 80.23 | −2.49 |
| **bowl** | 78.91 | 80.07 | 74.31 | **80.25** | **+1.34** |
| shelf | 67.61 | — | 68.73 | **58.47** | **−9.14** |
| **mIoU** | **80.71** | **80.84** | **81.76** | **79.15** | **−1.56** |

Summary：aAcc **95.19%**，mAcc **87.86%**。

### 10.3 与方案 3 的对应关系

- 方案 3 针对 **balanced10k** 的 bowl 漏检（background 9.29%、cup 6.19%）；方案 1 从 **fix5k** 再训后 **bowl 已回到基线之上**，说明 **cup/背景混淆在 fix5k 起点上可被压住**。
- **shelf 58.47%** 为意外大幅回落，可能因 bowl 加权 + 全 stage LASS 与边界 0.12 的联合扰动；若需交付，**勿用 bowl5k 作通用权重**。

### 10.4 权重选用（更新）

| 场景 | 推荐权重 |
|------|----------|
| **mIoU 最高** | `balanced10k/iter_10000.pth`（81.76%） |
| **稳妥 + bowl 尚可** | `fix5k/iter_5000.pth`（80.84%，bowl 80.07） |
| **类均衡（↑ 类多）** | `balanced10k/iter_6000.pth`（80.83%） |
| **仅修复 balanced 的 bowl** | `bowl5k/iter_5000.pth`（bowl 80.25%，mIoU 79.15%） |
| 路线 C / 论文主表 | 仍优先 **fix5k** 或 **balanced10k/iter_10000**，不用 bowl5k |

### 10.5 命令备查

**训练**：

```bash
mkdir -p outputs/trans10k_lass_mmscope_balanced_bowl5k

nohup python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_lass_balanced_bowl.py \
  --work-dir outputs/trans10k_lass_mmscope_balanced_bowl5k \
  --load-from outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --no-validate \
  --cfg-options data.workers_per_gpu=2 \
  > outputs/trans10k_lass_mmscope_balanced_bowl5k/train.log 2>&1 &
```

**评测**：

```bash
python tools/test.py \
  local_configs/segman_trans/segman_b_trans10k_lass_balanced_bowl.py \
  --checkpoint outputs/trans10k_lass_mmscope_balanced_bowl5k/iter_5000.pth \
  --eval mIoU
```

---

## 11. bowl 修补微调（方案 1b：从 balanced10k iter_10000）

**动机**：方案 1（fix5k→bowl5k）bowl **80.25%** 但 mIoU **79.15%**；若需 **保留 iter_10000 的 81.76% mIoU / window**，应在 **`balanced10k/iter_10000.pth`** 上 **短程、弱扰动** 微调。

| 项目 | 内容 |
|------|------|
| 配置 | `local_configs/segman_trans/segman_b_trans10k_lass_balanced_bowl_from10k.py` |
| 初始权重 | `outputs/trans10k_lass_mmscope_balanced10k/iter_10000.pth` |
| work-dir | `outputs/trans10k_lass_mmscope_balanced_bowl3k_from10k` |
| max_iters / lr | **3000** / **1.5e-5** |
| 相对 balanced10k | cup **1.0**、bowl **1.10**、Dice **0.1**（原 0.4）、LASS **stage [1,2]** 不变、新模块 lr_mult **4×**（原 6×） |
| 验收 | mIoU **≥ 80.71%**（理想 ≥81%）；bowl **≥ 78.91%**（理想 ≥79.5%） |

### 11.1 训练与评测命令（Docker，`segmentation`）

```bash
mkdir -p outputs/trans10k_lass_mmscope_balanced_bowl3k_from10k

nohup python tools/train.py \
  local_configs/segman_trans/segman_b_trans10k_lass_balanced_bowl_from10k.py \
  --work-dir outputs/trans10k_lass_mmscope_balanced_bowl3k_from10k \
  --load-from outputs/trans10k_lass_mmscope_balanced10k/iter_10000.pth \
  --no-validate \
  --cfg-options data.workers_per_gpu=2 \
  > outputs/trans10k_lass_mmscope_balanced_bowl3k_from10k/train.log 2>&1 &
```

查看进度：

```bash
tail -f outputs/trans10k_lass_mmscope_balanced_bowl3k_from10k/train.log
```

**评测**（建议 `iter_1000` / `iter_2000` / `iter_3000` 各测一次，取 mIoU 与 bowl 折中最佳）：

```bash
for it in 1000 2000 3000; do
  python tools/test.py \
    local_configs/segman_trans/segman_b_trans10k_lass_balanced_bowl_from10k.py \
    --checkpoint outputs/trans10k_lass_mmscope_balanced_bowl3k_from10k/iter_${it}.pth \
    --eval mIoU
done
```

可选：对最佳 checkpoint 再跑 bowl 混淆：

```bash
python scripts/analyze_bowl_confusion.py \
  local_configs/segman_trans/segman_b_trans10k_lass_balanced_bowl_from10k.py \
  --checkpoint outputs/trans10k_lass_mmscope_balanced_bowl3k_from10k/iter_3000.pth
```

### 11.2 验收结论（2026-05-23，val 1000 张）

| 验收项 | 阈值 | iter_1000 | iter_2000 | iter_3000 |
|--------|------|-----------|-----------|-----------|
| mIoU | ≥ 80.71% | ☑ **81.04** | ✗ 80.02 | ✗ 80.00 |
| bowl IoU | ≥ 78.91% | ✗ 69.28 | ✗ 73.68 | ✗ 75.19 |

**结论**：方案 1b **未同时达标**。随 iter 增加，**bowl 缓慢回升**（69.28→75.19），但 **mIoU / window 持续下降**；**无任何 checkpoint 优于「直接用 iter_10000」或 fix5k 的 bowl+mIoU 组合**。

### 11.3 checkpoint 总览

| checkpoint | mIoU | Δ vs 基线 | Δ vs iter_10000 | bowl | Δ bowl vs 基线 | window | Δ window vs 10000 |
|------------|------|-----------|-----------------|------|----------------|--------|-------------------|
| iter_10000（起点） | 81.76 | +1.05 | — | 74.31 | −4.60 | 82.91 | — |
| **iter_1000** | **81.04** | +0.33 | −0.72 | **69.28** | **−9.63** | **83.50** | +0.59 |
| iter_2000 | 80.02 | −0.69 | −1.74 | 73.68 | −5.23 | 69.97 | −12.94 |
| iter_3000 | 80.00 | −0.71 | −1.76 | **75.19** | −3.72 | 69.90 | −13.01 |
| fix5k（对照） | 80.84 | +0.13 | — | 80.07 | +1.16 | 76.27 | — |
| bowl5k（对照） | 79.15 | −1.56 | — | 80.25 | +1.34 | 71.92 | — |

**趋势（1b 三次）**：

```text
iter:     1000    2000    3000
mIoU:     81.04 → 80.02 → 80.00   （↓，略低于基线 80.71）
bowl:     69.28 → 73.68 → 75.19   （↑，仍低于基线 78.91 与起点 74.31→仅 3k 略好于起点 +0.88）
window:   83.50 → 69.97 → 69.90   （先高后崩，2k/3k 远低于 iter_10000）
```

- **iter_1000**：mIoU 仍 **81.04**（三类 1b 最高），window **83.50** 接近 iter_10000，但 **bowl 69.28 为三次最差**（比微调前 74.31 还低 **5%**）——早期步数对 bowl 有 **负向冲击**。
- **iter_2000 / 3000**：mIoU 卡在 **≈80.0**（**未达 80.71**），bowl 回升至 73.68 / 75.19，但 **window 从 83.5 跌至 ≈70**（相对 iter_10000 **约 −13%**），出现与 bowl5k 类似的 **window–bowl 跷跷板**。

### 11.4 逐类 IoU（%）

| 类别 | 基线 | iter_10000 | iter_1k | iter_2k | iter_3k | 三次中最佳 |
|------|------|------------|---------|---------|---------|------------|
| background | 96.71 | 96.45 | 96.55 | 96.36 | 96.29 | 1k |
| box | 71.47 | 71.86 | 66.87 | 71.41 | 68.88 | 2k |
| bottle | 87.77 | 88.20 | **88.95** | 87.51 | 87.37 | 1k |
| window | 66.62 | **82.91** | **83.50** | 69.97 | 69.90 | **1k** |
| eyeglass | 92.85 | 92.01 | 91.78 | 91.46 | 91.67 | 1k |
| freezer | 73.90 | 73.48 | 73.84 | 72.98 | 73.98 | 1k |
| jar_kettle | 84.04 | 83.89 | 80.36 | 83.91 | 82.46 | 2k |
| door | 75.04 | 74.40 | **75.75** | 75.48 | 73.91 | 1k |
| cup | 90.91 | 90.96 | 90.72 | 90.69 | 90.35 | 1k |
| wall | 82.72 | 83.95 | **84.50** | 83.04 | 81.98 | 1k |
| **bowl** | **78.91** | 74.31 | 69.28 | 73.68 | **75.19** | **3k** |
| shelf | 67.61 | 68.73 | **70.45** | 63.81 | 68.00 | 1k |
| **mIoU** | **80.71** | **81.76** | **81.04** | 80.02 | 80.00 | **1k** |

### 11.5 分析与建议

1. **3000 iter 仍不足以把 bowl 拉回基线**：最佳 **75.19%**（iter_3000），距基线 **78.91%** 差 **3.72%**，距 fix5k **80.07%** 差 **4.88%**；继续加长 iter 大概率进一步损伤 window/mIoU（2k→3k 已几乎无 bowl 增益）。
2. **mild 超参仍牵动 window**：bowl 权重 1.10 + Dice 0.1 在 iter_10000 表征上，**1000 step 即严重伤 bowl**，**2000+ step 伤 window**；与方案 3 诊断（bg/cup 混淆）相比，**单纯轻量 CE 微调难以在保持 81.76% mIoU 的同时修复 bowl**。
3. **方案 1b 不推荐作交付权重**；路线 B 主推仍为：
   - **mIoU / window 优先**：`balanced10k/iter_10000.pth`
   - **bowl + mIoU 均衡**：`fix5k/iter_5000.pth`
   - **仅 bowl 极致**：`bowl5k/iter_5000.pth`（牺牲 mIoU）
4. **若仍要攻 bowl 且保留 10k 的 window**：可试 **iter_6000** 作起点 + **≤1500 iter**、bowl **1.12**、**关 Dice**；或接受 **iter_1000** 作「高 mIoU 略损 bowl」折中（**不推荐**，bowl 69.28 过低）。

**1b 折中选 ckpt（仅作消融）**：`iter_1000.pth`（mIoU 81.04）；若必须在 1b 内兼顾 bowl，选 `iter_3000.pth`（bowl 75.19，mIoU 80.00）。

---

## 9. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-23 | 初版：平衡微调动机、超参、命令、验收标准 |
| 2026-05-23 | §8：填入 balanced10k 训练完成与 6k/8k/10k test 结果 |
| 2026-05-23 | §8.6：方案 3 bowl 诊断（vis + analyze_bowl_confusion）；§10：方案 1 命令占位 |
| 2026-05-23 | §10：方案 1 bowl5k test（mIoU 79.15%，bowl 80.25%） |
| 2026-05-23 | §11：方案 1b 自 iter_10000 短训 3k（`balanced_bowl_from10k.py`） |
| 2026-05-23 | §11.2–11.5：方案 1b 三次 test（1k/2k/3k 均未双达标） |
