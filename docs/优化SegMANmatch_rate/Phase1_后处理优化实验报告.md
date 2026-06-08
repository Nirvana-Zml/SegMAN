# Phase 1 后处理优化实验报告

| 项 | 内容 |
| --- | --- |
| 文档版本 | v1.0 |
| 执行日期 | 2026-06-08 |
| 关联计划 | [SegMAN_match_rate_提升实施计划.md](SegMAN_match_rate_提升实施计划.md) §3.1 |
| 执行环境 | Docker `segman_train`，conda `segman` |
| 目标模型 | `segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth` |
| 数据 | Trans10K val，1000 张 |
| 台账 | `outputs/match_improve/phase1_summary.json` |

---

## 1. 执行摘要

Phase 1 在 **不改权重** 的前提下，对语义轨 E2E 后处理参数进行系统 sweep。结论如下：

| 项 | 结果 |
| --- | --- |
| **主目标**（match ≥ 63%） | ❌ **未达成** |
| **最优 match_rate** | **60.84%**（`phase1_p1_iou_per_class`） |
| **相对 B1 基线提升** | **+1.68pp**（59.16% → 60.84%） |
| **deploy 决策** | **维持 B1**（`f1_b1_ref`，59.16%） |
| **瓶颈根因** | 实例 **漏检（miss）** 占未匹配 ~90%，CC/merge 无法根治 |

**务实判断**：后处理优化天花板约 **60.8%**，距 Phase 1 目标还差 **2.2pp**；若要继续提升，需进入 **Phase 2（弱类 finetune + 数据增强）**。

---

## 2. 目标与验收闸门

### 2.1 Phase 1 目标（计划 §3.1.9）

| 闸门 ID | 条件 | 实际值 | 结果 |
| --- | --- | --- | --- |
| P1-PASS-1 | match_rate ≥ **63.0%** | **60.84%** | ❌ |
| P1-PASS-2 | pred_gt_ratio ∈ [0.95, 1.10] | **1.039** | ✅ |
| P1-PASS-3 | wall_match ≥ **55%** | **50.62%** | ❌ |
| P1-PASS-4 | cls_on_matched 下降 ≤ 1.0pp | **84.86%**（+0.27pp） | ✅ |
| P1-PASS-5 | strict_e2e ≥ **52%** | **51.63%** | ❌ |

**综合判定**：`P1_PASS = false`，不更新正式 deploy 脚本。

### 2.2 B1 基线复现（Step 1）

| 指标 | 计划目标 | 实测 `phase1_p0_b1_baseline` |
| --- | --- | --- |
| match_rate | 59.16% ± 0.2pp | **59.16%** ✅ |
| pred_gt_ratio | 1.04 ± 0.02 | **1.046** ✅ |
| num_gt_instances | 3105 | **3105** ✅ |
| wall_match | ~47.9% | **47.91%** ✅ |
| door_match | ~45.4% | **45.40%** ✅ |

基线复现成功，后续所有 Δ 均相对此目录计算。

---

## 3. 实验设计与结果

### 3.1 执行脚本

```bash
# Docker 内全量 sweep（5 组 × 1000 张）
docker exec segman_train bash -lc '
  source /root/anaconda3/etc/profile.d/conda.sh &&
  conda activate segman &&
  cd /workspace/segman &&
  bash scripts/run_segman_match_p1_sweep.sh
'
```

脚本路径：`scripts/run_segman_match_p1_sweep.sh`

### 3.2 公共 B1 参数

```bash
--instance-source semantic
--min-area 128 --nms-iou 0.5 --max-aspect-ratio 10
--iou-match 0.25 --min-area-shelf 32
```

### 3.3 五组实验结果

| 实验 ID | 手段 | match_rate | Δ vs B1 | pred_gt_ratio | wall_match | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| `phase1_p0_b1_baseline` | B1 基线 | **59.16%** | — | 1.046 | 47.91% | 复现成功 |
| `phase1_p1_tta_default` | + TTA | **60.48%** | +1.32pp | 1.039 | 50.54% | ✅ 与历史 `d4_tta_on` 一致 |
| `phase1_p1_m1_dist_only` | TTA + 边距合并 | **60.48%** | +1.32pp | 1.039 | 50.54% | 边距合并无额外收益 |
| `phase1_p1_m2_conservative` | TTA + merge iou 0.08 | **43.67%** | −15.5pp | 0.666 | 25.50% | ❌ 过度合并灾难 |
| **`phase1_p1_iou_per_class`** | **TTA + per-class IoU** | **60.84%** | **+1.68pp** | 1.039 | 50.62% | ✅ **最优** |

### 3.4 最优参数（未上线，仅记录）

```bash
# 在 B1 基础上追加：
--seg-tta --seg-tta-scales 0.75,1.0,1.25
--iou-match-per-class door:0.22,wall:0.22,window:0.22,shelf:0.22
```

**代价**：TTA 使推理耗时约为基线 **6 倍**；收益 +1.68pp，仍远低于模式 B（75.46%）。

### 3.5 结构类 per-class 变化（最优 vs 基线）

| 类 | 基线 match | 最优 match | Δ |
| --- | --- | --- | --- |
| wall | 47.91% | 50.62% | **+2.71pp** |
| door | 45.40% | 47.96% | **+2.56pp** |
| window | 56.15% | 58.46% | **+2.31pp** |
| shelf | 53.23% | 54.84% | **+1.61pp** |

结构类均有提升，但 wall 仍远低于 55% 闸门，说明瓶颈在 **SegMAN 语义输出质量**，而非后处理阈值 alone。

---

## 4. 未匹配审计（Step 6）

对基线与最优配置执行 `export_unmatched_instances.py`：

| 指标 | 基线 | 最优 | 变化 |
| --- | --- | --- | --- |
| 未匹配 GT 总数 | 1268 | 1216 | **−52** |
| wall 未匹配 | 672 | 637 | −35 |
| door 未匹配 | 362 | 345 | −17 |
| 主因 miss 占比 | 88.0% | 90.0% | 仍占绝对主导 |

审计目录：

- 基线：`outputs/match_improve/phase1_audit_baseline/`
- 最优：`outputs/match_improve/phase1_audit_best/`

**解读**：后处理优化减少了少量漏检，但 **miss 仍是主因**（约 90%），CC 合并/阈值调整无法从根本上解决 wall/door 实例缺失。

---

## 5. 关键教训

### 5.1 已验证有效

1. **TTA**（`--seg-tta`，scales 0.75/1.0/1.25）：稳定 +1.3pp，历史可复现。
2. **per-class IoU 0.22**（在 TTA 之上）：额外 +0.36pp，合计 +1.68pp。
3. **B1 全参数**必须包含 `--max-aspect-ratio 10` 与 `--min-area-shelf 32`（`run_deliver_semantic_e2e.sh` 当前缺失这两项）。

### 5.2 禁止或无效

| 手段 | 结果 | 处置 |
| --- | --- | --- |
| `merge-cc-iou ≥ 0.08` | match 跌至 43.67% | **禁止** |
| `merge-cc-dist` only（wall,door） | 与 TTA 等价，无增益 | 不纳入 deploy |
| 形态学/膨胀/切分（历史 E/D 阶段） | 单因素负收益 | Phase 1 未重试，维持历史结论 |

### 5.3 与历史实验对照

| 历史 ID | Phase 1 对应 | 一致性 |
| --- | --- | --- |
| `d4_tta_on` | `phase1_p1_tta_default` | ✅ 60.48% 完全一致 |
| `b5_merge_wd` | `phase1_p1_m2_conservative` | ✅ 同样过度合并灾难 |
| `b3_per_class_iou` | `phase1_p1_iou_per_class` | 本次有小幅正收益（+0.36pp on TTA） |

---

## 6. Deploy 决策

| 项 | 决策 |
| --- | --- |
| 正式 deploy | **维持 B1**（59.16%，`outputs/e2e_improve/f1_b1_ref`） |
| `run_deliver_semantic_e2e.sh` | **不更新**（闸门未过） |
| 模式 B 抓取轨 | **不变**，继续推荐 M2F（75.46%） |
| Phase 1 最优参数 | 记入 manifest 与本文档，供可选语义轨或 Phase 2 基线 |

### 可选：非 deploy 语义增强

若仅需语义可视化小幅提升（接受 6× 推理耗时），可使用：

```bash
python transgrasp/pipelines/segment_and_classify.py \
  --image <path> --out-dir <out> \
  --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 \
  --iou-match 0.25 --min-area-shelf 32 \
  --seg-tta --seg-tta-scales 0.75,1.0,1.25 \
  --iou-match-per-class door:0.22,wall:0.22,window:0.22,shelf:0.22
```

---

## 7. 产出物清单

| 类型 | 路径 |
| --- | --- |
| 汇总台账 | `outputs/match_improve/phase1_summary.json` |
| 执行日志 | `outputs/match_improve/phase1_sweep.log` |
| 基线评测 | `outputs/match_improve/phase1_p0_b1_baseline/e2e_metrics_report.json` |
| 最优评测 | `outputs/match_improve/phase1_p1_iou_per_class/e2e_metrics_report.json` |
| 审计（基线） | `outputs/match_improve/phase1_audit_baseline/` |
| 审计（最优） | `outputs/match_improve/phase1_audit_best/` |
| sweep 脚本 | `scripts/run_segman_match_p1_sweep.sh` |
| 本报告 | `docs/优化SegMANmatch_rate/Phase1_后处理优化实验报告.md` |

---

## 8. 后续建议

1. **交付/比赛主线**：双轨定位不变——抓取用模式 B，语义用模式 A（B1）。
2. **若继续冲 match_rate**：启动 **Phase 2**（copypaste patch bank + 弱类两阶段 finetune），目标在 Phase 1 的 60.8% 基础上再 +3~4pp。
3. **文档收尾**（可选）：在 `viz_deliver/` 增加 Phase 1 五组实验对比图。
4. **修复 deploy 脚本**（低优先级）：为 `run_deliver_semantic_e2e.sh` 补全 `--max-aspect-ratio 10 --min-area-shelf 32`，与 B1 评测对齐（不改变当前 deploy 指标口径）。

---

**Phase 1 状态：已完成（PARTIAL_FAIL）**  
**下一步：Phase 2 弱类 finetune，或维持现状进入交付收尾。**
