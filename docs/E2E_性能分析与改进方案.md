# E2E 性能分析与改进方案

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.2 |
| 编写日期 | 2026-05-26 |
| 前置评测 | [E2E_segment_and_classify_测试说明.md](./E2E_segment_and_classify_测试说明.md) §10 |
| E2E 报告 | `outputs/e2e_segment_classify/val_full/e2e_metrics_report.md` |
| 关联文档 | 《OpenCLIP_细分类_未达80%原因与优化方案.md》§5.2 P0、《OpenCLIP_细分类_完整优化历程与交付说明.md》 |
| 当前 deliver | 分割 v2@6k `iter_6000.pth` + 分类 P3 `deliver_classifier_best.pth` |

---

## 1. 背景

2026-05-26 完成 **全量 val（1000 张）E2E 测试**：原图 → SegMAN 分割 → 连通域裁 ROI → P3 OpenCLIP 分类 → 按类拒识。  
此前优化（P0～P4、方案 B）主要用 **离线 ROI Top-1** 验收；E2E 首次用 **实例匹配 + 全 GT 严格 Acc** 度量真实流水线，暴露分割实例召回是主瓶颈。

**本文档目的**：基于 E2E 实测数据，给出 **分阶段、可验收、优先级明确** 的改进方案，与既有 P0 分割方案衔接，并补充 **实例后处理** 与 **修订验收口径**。

---

## 2. 现状性能画像

### 2.1 核心指标（val 1000 张 / 3105 GT 实例）

| 指标 | 数值 | 含义 |
|------|------|------|
| 实例匹配率 `match_rate` | **59.32%** | 预测 mask 与 GT 实例 IoU≥0.3 的比例 |
| E2E 分类 Top-1（匹配对上） | **84.09%** | 匹配成功子集上的细分类正确率 |
| E2E + 拒识（grasp only） | **90.83%** | 高置信自动决策正确率 |
| 分割语义类 Acc（匹配对上） | **92.94%** | 预测 mask 语义类 vs GT 类 |
| **严格端到端 Acc（全 GT）** | **≈49.9%** | 1549/3105，真实全链路召回 |
| 预测实例数 / GT 实例数 | 3563 / 3105 | 存在过分割 |
| **pred/GT 比** `pred_gt_ratio` | **1.148** | 3563÷3105；>1 表示冗余预测偏多 |
| **wall 未匹配实例数** | **≈669** | 1290×(1−48.14%)，全局最大漏检池 |
| **door 未匹配实例数** | **≈359** | 663×(1−45.85%) |

### 2.2 辅助指标（E1/E2 共用，基线必记）

除 match_rate 外，后续实验 **必须同时记录** 下列指标，避免「只刷匹配率、放任过分割」：

| 指标 | 基线 | 计算方式 |
|------|------|----------|
| `pred_gt_ratio` | **1.148** | 预测实例总数 ÷ GT 实例总数 |
| `redundancy_excess` | **458** | 预测实例总数 − GT 实例总数（粗粒度冗余量） |
| `redundancy_drop_rate` | — | `(baseline_excess − exp_excess) / baseline_excess`；E1 目标 **≥8%** |
| `unmatched_pred_ratio` | 待 E2-0 脚本统计 | 未匹配到任何 GT 的预测实例 ÷ 预测实例总数 |
| `cls_on_matched` | 84.09% | 匹配子集分类 Acc（不应为抬 match 牺牲 >2 pt） |

> `redundancy_excess` 为粗指标（未扣「该有但未检出」的 GT）；E2-0 后将补充 **按根因拆分的未匹配 GT 数**，用于指导 E2 训练侧重点。

### 2.3 与离线评测对照

| 评测 | Top-1 | 口径 |
|------|-------|------|
| GT-ROI 离线 | 76.91% | GT mask 裁 ROI，分类上界 |
| SegMAN-ROI 离线 | 67.49% | 预导出 v2@6k ROI |
| E2E 匹配对分类 | 84.09% | 仅 1842 个匹配 GT 实例（条件 Acc） |
| E2E 全 GT 严格 | ≈49.9% | 含未匹配实例 |

### 2.4 误差分解（误差预算）

```text
严格 E2E ≈ match_rate × cls_acc_on_matched
        ≈ 59.32% × 84.09% ≈ 49.9%

未匹配 GT 实例：3105 − 1842 = 1263（40.7%）→ 直接计 0 分
匹配但分类错：1842 × (1 − 84.09%) ≈ 293 个
```

| 误差来源 | 约占全 GT 错误 | 优先级 |
|----------|----------------|--------|
| **分割实例未匹配**（漏检/碎裂/IoU 不足） | **≈40.7%** | **P0** |
| **分类错误**（在已匹配实例上） | **≈9.4%** | P1 |
| 过分割 / 冗余预测 | 不直接进入严格 Acc 分母，但浪费算力、干扰匹配 | P0 |

**结论**：继续投入 **P4 式分类微调** 对严格 E2E 的边际收益低；**实例匹配率** 是首要杠杆。

---

## 3. 按类瓶颈诊断

### 3.1 三类问题模式

| 模式 | 代表类 | 匹配率 | 匹配后 Cls Acc | 改什么 |
|------|--------|--------|----------------|--------|
| **A 可交付强类** | cup, eyeglass, bottle | 88–94% | 88–100% | 维持 + 拒识 |
| **B 分割拖后腿、分类尚可** | **wall** | **48.1%** | **88.1%** | **分割实例召回** |
| **C 分割+分类双弱** | door, window, shelf | 46–58% | 50–69% | 分割优先，分类跟进 |

### 3.2 按类明细（E2E val_full）

| 类 | GT 实例 | 匹配率 | Cls Acc | Acc(grasp) | 全局影响 |
|----|---------|--------|---------|------------|----------|
| wall | **1290** | 48.14% | 88.08% | 94.07% | **最大**：占 GT 41%，半数未匹配 |
| door | **663** | 45.85% | 69.41% | 76.84% | **第二大**：数量多且双低 |
| cup | 366 | 94.26% | 96.23% | 97.34% | demo 主力 |
| eyeglass | 92 | 88.04% | 100% | 100% | demo 主力 |
| window | 130 | 58.46% | 59.21% | 71.15% | 结构混淆 |
| shelf | 62 | 54.84% | 50.00% | 63.64% | 小样本 + 分割差 |
| box | 88 | 64.77% | 78.95% | 86.49% | P0 审计分割落差大 |
| bottle / bowl / jar_kettle / freezer | 414 | 70–80% | 75–88% | 80–100% | 中等 |

### 3.3 与历史审计的一致性

`outputs/p0_weak_audit.md` 已指出：

- **door↔wall** 互混是 SegMAN-ROI 错分主体（wall→door 325，door→wall 255）。
- P0-1 弱类 finetune **mIoU 过闸门、SegMAN-ROI 反降**（61.81%），说明 **像素 IoU ≠ 实例召回**。
- 改进需同时看 **E2E match_rate** 与 **离线 SegMAN-ROI**，不能只看 mIoU。

---

## 4. 改进目标与修订验收口径

### 4.1 不建议再单一追求「GT-ROI Top-1 ≥ 80%」

P0～P4 已证：GT-ROI 实验最高 **77.04%**，deliver **76.91%**；继续同类分类训练性价比低。

### 4.2 建议采用「三层验收」

| 层级 | 指标 | 当前 | 阶段目标 | stretch |
|------|------|------|----------|----------|
| L1 分割 | mIoU | 81.80% | ≥81.5%（不回退） | ≥82% |
| **L2 E2E 实例** | **match_rate** | **59.32%** | **≥65%** | **≥70%** |
| **L2 E2E 实例** | **pred_gt_ratio** | **1.148** | **≤1.10** | **≤1.06** |
| **L2 E2E 实例** | **redundancy_drop_rate**（相对基线 excess） | 0% | **≥8%**（E1） | **≥15%** |
| **L2 E2E 实例** | **严格端到端 Acc** | **≈49.9%** | **≥55%** | **≥60%** |
| L2 按类 | wall 未匹配数 | ≈669 | **≤580**（≈55% 匹配率） | ≤520 |
| L2 按类 | door 未匹配数 | ≈359 | **≤298**（≈55% 匹配率） | ≤265 |
| L3 分类（匹配对） | cls_acc_on_matched | 84.09% | ≥84%（保持） | ≥86% |
| L3 部署决策 | grasp Acc | 90.83% | ≥88% | ≥92% |
| 参考离线 | SegMAN-ROI Top-1 | 67.49% | ≥68% | ≥70% |

**E2 wall/door 匹配率目标（52%/50%）的启用条件**：仅在 E2-0 根因审计完成后，按 **主导根因** 选用下表（§7.1），避免一刀切。

| 若 wall/door 未匹配主导根因 | wall 目标 | door 目标 | E2 训练侧重 |
|---------------------------|-----------|-----------|-------------|
| **漏检**（miss）占比 ≥40% | 50% | 48% | 类权重↑、Copy-Paste、难例重采样 |
| **碎裂**（fragment）占比 ≥40% | 52% | 50% | 边界 loss、连通域后处理、略增 min_area 类自适应 |
| **粘连**（adhesion）占比 ≥40% | 50% | 52% | **P0-4c door–wall boundary loss**、wall 权重略降 |
| **IoU 不足**（iou_gap）占比 ≥40% | 54% | 52% | E1 匹配阈值/NMS 微调 + 边界 refine，**非**优先重训 |

**答辩/demo 口径**：

- **全实例严格 Acc ≈50%** — 诚实反映全链路。
- **grasp 子集 91%** — 自动抓取安全决策。
- **cup/eyeglass/bottle** — 可演示「高可靠类」。

---

## 5. 改进路线图总览

```text
E1  实例后处理（1～2 天，低风险）     → 抬 match_rate，减过分割
E2  分割弱类 v2（3～5 天）            → wall/door 实例召回（P0 迭代 + 边界 loss）
E3  结构弱类分类（2～3 天，并行可选）  → window/door/shelf on SegMAN-ROI
E4  E2E 回归 + demo 策略（1 天）      → 固定闸门、类级开关
──  不建议 ──
X   P4-full / 换 backbone 冲 GT 80%  → 对 E2E 49.9% 帮助间接且慢
X   P3+P2 级联                        → 已证 GT 75.27% < P3 单头
```

**预期收益（粗算，分类 Acc 不变）**：

| 若 match_rate 提升至 | 严格 E2E（×84% cls） |
|---------------------|----------------------|
| 65% | ≈54.6% |
| 70% | ≈58.8% |

---

## 6. 阶段 E1 — 实例后处理（优先执行）

**动机**：3563 预测实例 > 3105 GT，存在噪声 CC 与重复框；**不改模型权重**，只改 ROI 提取逻辑。

### E1-1 连通域过滤增强

| 项 | 现状 | 建议 |
|----|------|------|
| `min_area` | 64 px | val 上 sweep **128 / 256** |
| 宽高比 / 极端细长 | 无 | 过滤 aspect ratio > 10 的 CC |
| 同类 NMS | 无 | 同 class_id IoU>0.5 的 bbox 保留面积最大 |

**实现位置**：`transgrasp/pipelines/roi_extract.py`（新增 `--postprocess` 或 config JSON）。

### E1-2 匹配策略优化

| 项 | 现状 | 建议 |
|----|------|------|
| IoU 阈值 | 0.3 | sweep **0.25 / 0.35** 看 match_rate vs 误配 |
| 匹配算法 | 贪心 per GT | 可选匈牙利算法（bi-partite max IoU） |

**实现位置**：`transgrasp/pipelines/segment_and_classify.py` → `match_instances_to_gt()`。

### E1-3 验收（公平 GT 口径）

**强制约定**：GT 实例提取固定 `min_area=64`、无 NMS（`build_gt_extract_config`）；仅对 **预测 mask** 施加下表后处理。

```bash
bash scripts/run_e2e_regression.sh
# 或指定参数：
# bash scripts/run_e2e_regression.sh --min-area 128 --nms-iou 0.5 --iou-match 0.25 --min-area-shelf 32
```

| 闸门 | 条件 | 说明 |
|------|------|------|
| **E1-PASS（双指标）** | ① `match_rate` ≥ **62%**（+2.7 pt vs 基线）<br>② `redundancy_drop_rate` ≥ **8%**（excess 458→≤421，或 `pred_gt_ratio` ≤ **1.10**）<br>③ `e2e_top1_on_matched` ≥ **83%** | **必须同时满足**；禁止仅通过压低 pred 数「刷」匹配率 |
| E1-STRETCH | ①≥63% 且 ②≥12% 且 ③≥83.5% | 可合并入 E1-002 默认配置 |
| E1-FAIL | match 升但 ② 不满足（pred_gt_ratio 仍 >1.12） | 过分割未控 → 回退 NMS/min_area |
| E1-FAIL | match 升但 cls_on_matched 降 **>2 pt** | 过滤过激丢真实例 → 回退参数 |

**E1 阶段按类失败案例（必记）**：

| 类 | 基线未匹配数 | E1 期望减少 | 备注 |
|----|-------------|-------------|------|
| wall | ≈669 | **≥40**（→≤629） | 以后处理为主，难大幅解决粘连 |
| door | ≈359 | **≥20**（→≤339） | |
| shelf | ≈28 | **≥5** | 小实例；可试 per-class min_area |

**工期**：1～2 天。**风险**：低。

---

## 7. 阶段 E2 — 分割弱类专项 v2（核心）

**动机**：P0-1 仅看 mIoU 未改善 SegMAN-ROI；E2E 证明 **wall/door 实例匹配率 <50%** 是主矛盾。  
**前置依赖**：E2-0 根因审计完成后再定 E2-1 超参与 wall/door 分项目标（§4.2 表）。

### 7.1 未匹配根因 taxonomy（本项目统一定义）

对 **每个未匹配的 GT 实例**，在审计时归入 **唯一主导类**（优先级：miss > adhesion > fragment > iou_gap > class_swap）：

| 根因代码 | 判定条件（相对 GT 实例） | 典型目视 | 优先手段 |
|----------|------------------------|----------|----------|
| **miss** 漏检 | 预测 mask 在该 GT 区域 max IoU **<0.10** | 该物体区域全为背景/它类 | 类权重↑、Copy-Paste、难例 mining |
| **adhesion** 粘连 | 有 overlap 但 pred 为 **door↔wall** 合并 CC，或边界贴邻导致 CC 合并 | door 与 wall 连成一片 | **P0-4c boundary loss**、wall 权重略降 |
| **fragment** 碎裂 | 同一 GT 对应 **≥2 个 pred CC**，且每个 IoU 均 **<0.30** | 大 wall 被撕成多块 | 边界 loss、CRF/后处理 merge、训练时 dilate 一致性 |
| **iou_gap** IoU 不足 | 单 pred CC，语义类 **正确**，IoU **∈[0.10, 0.30)** | mask 偏移、略缩小 | E1 匹配阈值、边界 refine；**非**首选重训 |
| **class_swap** 语义错类 | IoU 最高 pred 的 **class_id ≠ GT**，且 IoU≥0.10 | window 被标成 wall | 类权重、hard negative、E3 分类 |

### 7.2 E2-0 审计（1～1.5 天，必做）

#### 7.2.1 自动导出未匹配清单

从 `val_full/per_image/*.json` 提取 `eval.matches` 中 `matched=false` 的 GT 实例，输出：

```text
outputs/e2e_improve/e2_audit/
├── unmatched_gt_instances.csv      # stem, gt_class, bbox, best_iou, best_pred_class
├── unmatched_by_class.json         # 按类计数
└── candidate_stems_wall_door.txt   # 供抽样标注
```

建议脚本（待建）：`transgrasp/pipelines/export_unmatched_instances.py`

```bash
python transgrasp/pipelines/export_unmatched_instances.py \
  --eval-dir outputs/e2e_segment_classify/val_full \
  --out-dir outputs/e2e_improve/e2_audit
```

#### 7.2.2 可视化标注流程（人工，抽样）

| 步骤 | 操作 | 产出 |
|------|------|------|
| 1 | 从 `unmatched_gt_instances.csv` **分层抽样**：wall **100**、door **50**、其余弱类各 **10** | `sample_list.csv` |
| 2 | 脚本生成三联图：`原图 \| GT mask 叠加 \| Pred mask 叠加`，标 GT bbox | `e2_audit/vis/{stem}_{gt_class}.jpg` |
| 3 | 两人交叉标注主导根因（§7.1 五选一） | `e2_audit/annotations.csv` |
| 4 | 统计根因占比，写入审计报告 | `e2_audit_unmatched.md` |

`annotations.csv` 列：`stem, gt_class, root_cause, note, annotator`

**质检**：wall+door 样本中至少 **20%** 双标一致；不一致条目第三人仲裁。

#### 7.2.3 问题类型 → 优化手段映射表（示例，E2-0 后替换为实测占比）

| 问题类型 | 典型类 | E2-0 若占比≥40% 则 | 对应阶段 |
|----------|--------|---------------------|----------|
| door–wall **粘连** | door, wall | 启用 **P0-4c boundary loss**；wall CE×0.95 | E2-1 |
| **碎裂** | wall, window | 边界 loss + pred CC **merge** 后处理（E1 扩展） | E1→E2 |
| **漏检**（小实例） | shelf, box, freezer | shelf/box **class_weight↑**；**per-class min_area**（shelf 可降至 32） | E2-1 + E1 |
| **IoU 不足** | 全类 | 优先 E1 `--iou-match` 0.25 试验；分割侧轻量 boundary refine | E1 |
| **语义错类** window↔wall | window | 维持 E2 分割 + **E3** window hard mining | E3 |

#### 7.2.4 E2-0 交付物

| 文件 | 内容 |
|------|------|
| `e2_audit_unmatched.md` | 未匹配总数、按类、**按根因占比**（wall/door 分表） |
| `e2_root_cause_matrix.csv` | 类 × 根因 交叉表 |
| `e2_action_decision.md` | **一页决策**：E2-1 开哪些 loss/权重/iter |
| `e2_audit/vis/` | 抽样可视化 |

**E2-0 完成标准**：wall、door 根因占比各已填，且 `e2_action_decision.md` 明确 E2-1 **最多 2 项**主改动（避免 P0-1 式多变量同时改）。

### 7.3 E2-1 分割训练（P0 迭代，2～4 天）

**仅执行 `e2_action_decision.md` 中勾选的改动**；默认候选池：

| 变更 | 目的 |
|------|------|
| **P0-4c door–wall boundary loss** | 减 door↔wall 粘连（见《未达80%》§5.2 P0-4c） |
| shelf/door/box **class_weight↑**，wall **↓0.95** | 延续 P0 审计策略 |
| **max_iters 2000**，lr **5e-6**（更保守） | 避免 P0-1 过拟合 val |
| 早停：看 **weak 类 IoU 均值 + E2E match_rate** | 不单看 mIoU |

**训练命令（Docker）**：

```bash
cd /workspace/segman/segmentation
python train.py local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2_e2weak.py \
  --work-dir outputs/trans10k_lass_mmscope_balanced_v2_e2weak
```

（需新建 config `segman_b_trans10k_lass_balanced_v2_e2weak.py`，自 v2@6k 热启动。）

### 7.4 E2-2 E2E 验收（必做，替代仅跑 mIoU）

```bash
python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2_e2weak/<CKPT>.pth \
  --out-dir outputs/e2e_improve/e2_seg_<CKPT>

python transgrasp/pipelines/summarize_e2e_eval.py \
  --eval-dir outputs/e2e_improve/e2_seg_<CKPT>
```

| 闸门 | 条件 |
|------|------|
| E2-PASS | mIoU ≥ **81.0%** 且 **match_rate ≥ 65%** 且 **pred_gt_ratio ≤ 1.08** |
| E2-PASS（按类） | wall/door 匹配率 ≥ **E2-0 决策表目标**（§4.2，通常 50–54%） |
| E2-PASS（按类） | wall 未匹配数 **减少 ≥80**（669→≤589）；door 未匹配数 **减少 ≥40**（359→≤319） |
| E2-PASS（根因） | 相对 E2-0 基线，**主导根因**对应未匹配数下降 **≥15%**（如 adhesion 类 −15%） |
| E2-KEEP | 严格 E2E ≥ **55%** → 可替换 deliver 分割权重 |
| E2-FAIL | SegMAN-ROI 离线 < **66%** 或 mIoU < **80.5%** → 回退 v2@6k |

### 7.5 E2-3 离线 SegMAN-ROI 对照（辅助）

固定 P3 分类 ckpt，重导 ROI 并评测（与 P0-2 流程相同）：

```bash
bash scripts/run_p0_remaining.sh   # 改 P0_DIR / CFG 指向 e2weak
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint outputs/openclip_classifier/deliver_classifier_best.pth \
  --roi-root data/trans10k_roi_segman_e2weak \
  --split val \
  --report-dir outputs/openclip_classifier/e2weak_eval_segman_roi
```

**工期**：3～5 天。**风险**：中（P0-1 曾 FAIL，需保守 schedule + E2E 闸门）。

---

## 8. 阶段 E3 — 结构弱类分类（次要，可与 E2 并行）

**动机**：door/window/shelf 在 **已匹配** 实例上 Cls Acc 仍低（50–69%）。

### E3-1 数据

- 用 E2 新分割导出的 `trans10k_roi_segman_e2weak/val`（或现网 v2@6k ROI）。
- 仅增广 **door / window / shelf / wall** hard pairs（延续 P3 hard mining 思路）。

### E3-2 训练

| 项 | 建议 |
|----|------|
| 基座 | P3 `p3_p1_hardmining/best.pth` |
| 数据 | SegMAN-ROI + `--aug p3` + 结构类 sample_weight×2 |
| unfreeze | 保持 4 block，lr **5e-6**，**≤3 epoch** |
| 早停 | 看 **E2E 上 door/window 匹配后 Acc**，非 GT-ROI |

### E3-3 闸门

| 条件 | 阈值 |
|------|------|
| GT-ROI | 不得 < **76.5%**（不回退 deliver） |
| E2E cls on matched (door) | ≥ **74%** |
| E2E cls on matched (window) | ≥ **65%** |

**工期**：2～3 天。**风险**：低～中（过拟合 GT 风险，必须以 SegMAN/E2E 为准）。

---

## 9. 阶段 E4 — E2E 回归与 demo 策略

### E4-1 固定回归脚本

新建 `scripts/run_e2e_regression.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/e2e_segment_classify/regression_$(date +%Y%m%d)
python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --seg-checkpoint "${SEG_CKPT:-segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth}" \
  --cls-checkpoint "${CLS_CKPT:-outputs/openclip_classifier/deliver_classifier_best.pth}" \
  --out-dir "${OUT}"
python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${OUT}"
```

每次改分割/分类/后处理后 **必须跑全量 val** 对比 `val_full` 基线。

### E4-2 Demo 类级策略

| 类 | 策略 |
|----|------|
| cup, eyeglass, bottle | 默认 **auto grasp**（匹配率>78%） |
| bowl, jar_kettle, box | grasp + 略高 τ |
| door, wall, window, shelf | **默认 reject 或人工确认**；展示 E2E 诚实边界 |

配置文件可扩展：`transgrasp/classification/configs/demo_class_policy.json`（`auto` / `confirm` / `disabled`）。

### E4-3 文档同步

改进完成后更新：

- `docs/E2E_segment_and_classify_测试说明.md` §10
- `OpenCLIP_细分类_完整优化历程与交付说明.md`
- `deliver_experiment_best/metrics_summary.md`

---

## 10. 不建议投入的方向

| 方向 | 原因 |
|------|------|
| P4-full / 更大 contrastive | P4-full 76.91%，未超 P3；对 E2E match_rate 无直接帮助 |
| 换 ViT-L / SigLIP 冲 GT 80% | 成本高；E2E 瓶颈在分割实例 59% 匹配率 |
| P3+P2 级联 | 已测 GT 75.27% < P3 单头 76.91% |
| 仅提高 mIoU 验收 | P0-1 教训：mIoU 81% 但 ROI Acc 下降 |
| 收紧 ROI bbox-pad | §9 实验已证无效甚至负向 |

---

## 11. 实验记录与闸门汇总

### 11.1 基线（2026-05-26，不可删）

| 键 | 值 |
|----|-----|
| 目录 | `outputs/e2e_segment_classify/val_full/` |
| match_rate | 59.32% |
| pred_gt_ratio | 1.148 |
| redundancy_excess | 458 |
| e2e_top1_on_matched | 84.09% |
| e2e_top1_grasp_only | 90.83% |
| strict_e2e_all_gt | ≈49.9% |
| wall 未匹配数 | ≈669 |
| door 未匹配数 | ≈359 |
| E2-0 根因占比 | miss **81.9%**（见 `e2_audit_baseline/`） |

### 11.2 实验台账（2026-05-26 执行）

**公平 GT=3105**；详见 `outputs/e2e_improve/e2_execution_summary.md`。

| 实验 ID | 内容 | match_rate | pred_gt_ratio | redundancy_drop | strict E2E | grasp Acc | wall 未匹配 Δ | door 未匹配 Δ | E1_PASS |
|---------|------|------------|---------------|-----------------|------------|-----------|---------------|---------------|---------|
| baseline | val_full | 59.32% | 1.148 | — | 49.9% | 90.83% | — | — | — |
| E1-001 | min_area=128 | 58.20% | 1.032 | 78.2% | 49.3% | 84.67% | −18（669→687） | −12 | ❌ |
| E1-002 | +NMS | 58.20% | 1.032 | 78.2% | 49.3% | 84.67% | −18 | −12 | ❌ |
| E1-003-fair | +iou0.25+shelf32 | **59.16%** | **1.046** | **69.0%** | 49.2% | 84.59% | −3（669→672） | −3 | ❌ |
| E1-iou025 | 仅 iou=0.25 | 59.16% | 1.122 | 17.5% | 49.2% | 84.59% | −3 | −3 | ❌ |
| E2-0 | 根因审计 | — | — | — | — | — | miss 82% | miss 82% | → E2-1 |
| E2-001 | e2weak 训练 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 目标 −80 | 目标 −40 | 待跑 |

**E1 执行结论**：公平口径下 **未达 match≥62%**；**推荐采纳 E1-003-fair 参数减冗余**（excess 458→142），match 与基线持平。**抬 match 须进入 E2-1 分割训练**。

**`wall 未匹配 Δ` 填写示例**：`−52（669→617）` 表示 wall 未匹配减少 52 个。  
**`主导根因变化` 填写示例**：`adhesion 38%→31%（−15% 相对）` — 需 E2-0 与实验后各做一次同规模抽样标注对比（可选，E2 必做 wall/door 各 30 样本快审）。

### 11.3 推荐执行顺序

```text
Week 1
  Day 1–2   E1 实例后处理 → 双指标：match≥62% 且 redundancy_drop≥8%
  Day 3–4   E2-0 未匹配导出 + 抽样标注 + 根因矩阵 + e2_action_decision.md
  Day 5–7   E2-1 分割 e2weak（仅 1～2 项主改动）+ E2E 验收

Week 2（可选）
  Day 1–3   E3 结构弱类分类（若 E2 match≥65% 后 cls 仍瓶颈）
  Day 4     E4 回归 + demo policy + 更新 §11.1 基线对比
```

---

## 12. 相关文件索引

| 内容 | 路径 |
|------|------|
| E2E 主脚本 | `transgrasp/pipelines/segment_and_classify.py` |
| ROI 提取 | `transgrasp/pipelines/roi_extract.py` |
| E2E 汇总 | `transgrasp/pipelines/summarize_e2e_eval.py` |
| 未匹配导出 | `transgrasp/pipelines/export_unmatched_instances.py` |
| E1 闸门检查 | `transgrasp/pipelines/check_e1_gates.py` |
| 执行摘要 | `outputs/e2e_improve/e2_execution_summary.md` |
| 一键 E1 公平重跑 | `scripts/run_e1_rerun_fair.sh` |
| E2-1 训练 | `scripts/run_e2_e2weak_train.sh` |
| E2-2 E2E 评测 | `scripts/run_e2e_e2_eval.sh` |
| E4 回归 | `scripts/run_e2e_regression.sh` |
| 拒识阈值 | `transgrasp/classification/configs/reject_thresholds_p3.json` |
| E2E 基线报告 | `outputs/e2e_segment_classify/val_full/e2e_metrics_report.md` |
| P0 弱类审计 | `outputs/p0_weak_audit.md` |
| 分类 deliver | `outputs/openclip_classifier/deliver_classifier_best.pth` |
| 分割 deliver | `segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth` |
| E2E 测试说明 | `docs/E2E_segment_and_classify_测试说明.md` |

---

## 13. 一句话总结

**当前模型「裁对之后分得好」（匹配对 84%，grasp 91%），「找不全实例」（匹配率 59%）**；改进应 **优先 E1 后处理 + E2 wall/door 分割实例召回**，分类侧 **维持 P3 deliver + 拒识**，结构弱类 **E3 小步跟进**；验收以 **E2E match_rate 与严格端到端 Acc** 为主，不再单一追求 GT-ROI 80%。

---

## 修订记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-05-26 | v1.0 | 基于 val_full E2E 实测撰写；E1～E4 分阶段方案与闸门 |
| 2026-05-26 | v1.1 | 精细化 E1 双指标闸门；E2-0 根因 taxonomy + 可视化标注流程；实验台账增加失败案例 Δ |
| 2026-05-26 | v1.2 | **执行 E1+E2-0**：公平口径 E1 未过 62% match；E2-0 miss 主导→推进 E2-1 |
