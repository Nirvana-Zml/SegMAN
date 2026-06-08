# SegMAN v2@6k match_rate 提升实施计划

**文档版本**：v1.2  
**创建日期**：2026-06-02  
**Phase 1 完成**：2026-06-08 → 见 [Phase1_后处理优化实验报告.md](Phase1_后处理优化实验报告.md)  
**目标模型**：`segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth`  
**当前 deploy 基准**：Trans10K val 语义轨 E2E match_rate **59.16%**（`f1_b1_ref`，维持不变）  
**Phase 1 最优（未上线）**：**60.84%**（`phase1_p1_iou_per_class`，+1.68pp）  
**参考对比**：M2F 抓取轨 **75.46%**（+16.3pp）  
**核心瓶颈类**：wall（47.9%）、door（45.4%）、window（56.2%）、shelf（53.2%）

### Phase 1 执行状态（2026-06-08）

| 项 | 结果 |
| --- | --- |
| 状态 | **PARTIAL_FAIL**（未达 63% 主目标） |
| 最优实验 | `phase1_p1_iou_per_class`：TTA + per-class IoU 0.22 |
| deploy | 维持 B1，不更新 `run_deliver_semantic_e2e.sh` |
| 台账 | `outputs/match_improve/phase1_summary.json` |
| 下一步 | 按 §3.2 执行 Phase 2（弱类 finetune + copypaste），或维持双轨交付现状 |

---

## 1. 背景与目标

### 1.1 问题陈述
- SegMAN v2@6k 在像素级 mIoU 达到 81.80%，但实例级 match_rate（IoU≥0.25）仅 59.16%。
- 主要拖累来自大面积透明结构类（wall/door/window/shelf），语义连通域（CC）易出现粘连、碎裂、漏检。
- 项目已决定将模式 B（M2F）作为抓取主轨，但仍希望在语义轨上挖掘潜力，用于：
  - 语义可视化 / 标注辅助
  - 离线 SegMAN-ROI 评测
  - 作为模式 B 的备选或融合输入

### 1.2 目标（分阶段）
| 阶段 | 时间 | 目标 match_rate | 验收条件 |
|------|------|------------------|----------|
| Phase 0（现状） | - | 59.16% | `f1_b1_ref` |
| Phase 1（低成本） | 1~2 周 | **≥ 63%** | 仅改后处理参数，无需重训 |
| Phase 2（中成本） | 3~4 周 | **≥ 66%** | 弱类 finetune + 数据增强 |
| Phase 3（可选） | 视情况 | ≥ 70% | 架构级改动（不推荐优先） |

**成功标准**：Phase 1 至少 +4pp；Phase 2 至少 +7pp；同时保持 SegMAN-ROI 分类 Acc 不显著下降（< 3pp）。

---

## 2. 现状诊断（数据驱动）

### 2.1 关键指标来源
- E2E 评测：`outputs/e2e_improve/f1_b1_ref/e2e_metrics_report.json`
- Per-class match：`viz_deliver/output/02_per_class_match_rate_meta.json`
- 混淆与 P/R/F1：`viz_deliver/output/04_confusion_matrix_meta.json` + `05_per_class_prf_meta.json`
- Coverage–Accuracy：`plan_b/coverage_gt/coverage_accuracy.json`

### 2.2 主要发现
1. **实例漏检为主，过检为辅**：pred_gt_ratio ≈ 1.045，冗余不多。
2. **结构类三角混淆**：door↔wall、window/shelf 被吸入 wall（混淆矩阵）。
3. **CC 参数已接近天花板**：E1/E2 阶段大量 post-processing 消融，增益有限。
4. **P0 弱类 finetune 失败教训**：仅调 loss weight 导致 SegMAN-ROI Acc 下降，需数据 + 训练策略双管齐下。

---

## 3. 改进方案分层实施

### Phase 1：低成本后处理优化（详细步骤）

**目标**：不改 `iter_6000.pth`，仅调 `segment_and_classify.py` 已有 CLI 参数，将 match_rate 从 **59.16%** 提升到 **≥63%**（+3.8pp）。

**时间**：1～2 周（含基线复现、单因素 sweep、组合验证、未匹配审计）。

---

#### 3.1.0 代码链路（必须先理解）

语义轨 E2E 在 `transgrasp/pipelines/segment_and_classify.py` 的 `process_one_image()` 中执行：

```text
RGB
  → SegMANSegmentor.predict_label_map()     # seg_model.py，iter_6000
  → [可选] apply_seg_refine()               # seg_refine.py（D 方案：形态学/膨胀/CRF）
  → extract_instance_rois()                 # roi_extract.py：每类 cv2.connectedComponents
  → postprocess_instances()               # roi_postprocess.py：filter → merge → NMS
  → classify_instances()                    # 不影响 match，只影响 cls
  → match_instances_to_gt()                 # greedy/hungarian，mask IoU ≥ thresh
```

**关键实现细节（影响调参方向）：**

| 模块 | 文件 | 行为 |
| --- | --- | --- |
| CC 提取 | `roi_extract.py` L60-91 | 对 11 类前景逐类 `connectedComponents(connectivity=8)` |
| 面积过滤 | `roi_postprocess.filter_instances()` | `min_area` 默认 64；`min_area_per_class` 仅 shelf 有 CLI |
| CC 合并 | `roi_postprocess.merge_nearby_cc()` | **NMS 之前**执行；同类 bbox IoU ≥ merge_iou **或** 边距 ≤ merge_dist 则合并 mask |
| NMS | `roi_postprocess.nms_instances()` | 同类 bbox IoU ≥ nms_iou 时抑制小实例 |
| GT 侧 | `build_gt_extract_config()` | 固定 `min_area=64`、无 NMS，保证 match 评测公平 |
| 匹配 | `match_instances_greedy()` | 每 GT 取 IoU 最大且未占用的 pred；阈值可用 `--iou-match-per-class` |

**结论**：match_rate 只受 **pred 实例数/形状** 影响，与 OpenCLIP 分类无关；调参应聚焦 SegMAN 输出 → CC → 后处理 → 匹配阈值。

---

#### 3.1.1 历史实验教训（避免重复踩坑）

以下结果均来自 `outputs/e2e_improve/`，基线 B1 = `b5_no_merge` / `f1_b1_ref`（**match 59.16%**）：

| 实验 ID | 手段 | match_rate | Δ | 结论 |
| --- | --- | --- | --- | --- |
| `e1_003_full_fair` | min128+NMS+shelf32+iou0.25 | 59.16% | 0 | **当前 deploy 基线** |
| `d4_tta_on` | B1 + `--seg-tta` | **60.48%** | **+1.32pp** | ✅ **目前最有效单因素** |
| `b3_per_class_iou` | 按类 IoU 阈值 | 59.10% | −0.06pp | 收益极小 |
| `b2_hungarian` | 匈牙利匹配 | 59.16% | 0 | 与 greedy 等价 |
| `d1_morph` | morph_close=5 | 58.23% | −0.93pp | ❌ 单独启用反而降 |
| `d2_dilate_wall1_door1` | wall/door 膨胀 1px | 58.29% | −0.87pp | ❌ 单独启用反而降 |
| `d5_split_dw` | door-wall 切分 | 58.39% | −0.77pp | ❌ 优先级低 |
| **`b5_merge_wd`** | **merge_iou=0.3, wall+door** | **44.99%** | **−14.2pp** | ❌ **灾难：过度合并导致漏检** |

**Phase 1 原则：**

1. **禁止**在未 sweep 的情况下使用 `merge-cc-iou ≥ 0.25`。
2. **优先**复现并固化 `d4_tta_on`（+1.3pp 已验证）。
3. 形态学/膨胀仅在 **TTA 基线之上** 小步尝试，且 kernel 要小（3～5）。
4. CC 合并优先用 **`merge-cc-dist`（边距合并）** 而非高 IoU 合并。

---

#### 3.1.2 Step 0 — 环境与前置检查（0.5 天）

**环境**：Docker `segman_train` 或本地 `conda activate segman`。

```bash
cd /workspace/segman   # 或 D:\SegMAN-main\SegMAN

# 1) 权重存在
test -f segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth
test -f outputs/openclip_classifier/deliver_classifier_best.pth

# 2) 数据存在
test -d segmentation/data/trans10k/img_dir/val
test -d segmentation/data/trans10k/ann_dir/val

# 3) 冒烟（1 张图，约 30s）
python transgrasp/pipelines/segment_and_classify.py \
  --image segmentation/data/trans10k/img_dir/val/val_000000.jpg \
  --out-dir outputs/match_improve/phase1_smoke \
  --min-area 128 --nms-iou 0.5 --iou-match 0.25 --min-area-shelf 32 \
  --save-sem-seg --save-rois
```

**通过标准**：`outputs/match_improve/phase1_smoke/summary.json` 生成且无报错。

---

#### 3.1.3 Step 1 — 锁定 B1 基线（0.5 天）

**目的**：确认本机可复现 `f1_b1_ref` 的 **59.16%**，后续所有 Δ 相对此目录。

**B1 参数（与 `scripts/run_e2e_regression.sh` 默认一致）：**

```bash
--instance-source semantic \
--min-area 128 \
--nms-iou 0.5 \
--max-aspect-ratio 10 \
--iou-match 0.25 \
--min-area-shelf 32
```

**全量 val 命令：**

```bash
OUT=outputs/match_improve/phase1_p0_b1_baseline

python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --instance-source semantic \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --cls-checkpoint outputs/openclip_classifier/deliver_classifier_best.pth \
  --out-dir "${OUT}" \
  --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 \
  --iou-match 0.25 --min-area-shelf 32

python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${OUT}"
python transgrasp/pipelines/check_e1_gates.py --eval-dir "${OUT}"
```

**验收（对齐 `f1_b1_ref/e2e_metrics_report.json`）：**

| 指标 | 目标 |
| --- | --- |
| match_rate | **59.16% ± 0.2pp** |
| pred_gt_ratio | **1.04 ± 0.02** |
| num_gt_instances | 3105 |
| wall match | ~47.9% |
| door match | ~45.4% |

未达标则先排查：checkpoint 路径、`--max-aspect-ratio` / `--min-area-shelf` 是否遗漏（`run_deliver_semantic_e2e.sh` 当前**未含**这两项，Phase 1 必须以 B1 全参数为准）。

---

#### 3.1.4 Step 2 — TTA（D4，最高优先级，1 天）

**代码**：`seg_model.py` → `predict_label_map_tta()`；scales × flip 众数投票。  
**历史**：`d4_tta_on` 已达 **60.48%**（wall 47.9%→50.5%）。

**2a. 复现历史最优：**

```bash
OUT=outputs/match_improve/phase1_p1_tta_default

python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --instance-source semantic \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --cls-checkpoint outputs/openclip_classifier/deliver_classifier_best.pth \
  --out-dir "${OUT}" \
  --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 \
  --iou-match 0.25 --min-area-shelf 32 \
  --seg-tta --seg-tta-scales 0.75,1.0,1.25

python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${OUT}"
```

**2b. 可选 scale sweep（子集 100 张加速）：**

| 实验 ID | `--seg-tta-scales` |
| --- | --- |
| p1_tta_s1 | `1.0`（仅 flip，需改代码或 scales=1.0） |
| p1_tta_s2 | `0.75,1.0,1.25`（默认） |
| p1_tta_s3 | `0.9,1.0,1.1`（轻量 TTA，耗时更低） |

**验收**：match ≥ **60.3%**；若 < 60.0% 检查 GPU/权重是否与 `d4_tta_on` 一致。  
**代价**：推理耗时约为基线 **6 倍**；生产部署需权衡 latency。

---

#### 3.1.5 Step 3 — 保守 CC 合并 sweep（1～2 天）

**代码**：`roi_postprocess.merge_nearby_cc()`，在 `postprocess_instances()` 中 **NMS 前**调用。

**严禁**：`merge-cc-iou 0.3`（已证 match 跌至 44.99%）。

**在 Step 2 最优 TTA 配置上 sweep**（若无 TTA 提升则退回 B1）：

| 实验 ID | merge-cc-iou | merge-cc-dist | merge-cc-classes | 说明 |
| --- | --- | --- | --- | --- |
| p1_m1 | 0 | 12 | wall,door | 仅边距合并，不用 IoU |
| p1_m2 | 0.08 | 12 | wall,door,window | 极低 IoU + 边距 |
| p1_m3 | 0.10 | 16 | wall,door,window,shelf | 略放宽边距 |
| p1_m4 | 0.15 | 12 | wall,door | 原计划上限，需严防 pred 数骤降 |

**单次命令模板：**

```bash
OUT=outputs/match_improve/phase1_p1_m2
# ... 继承 Step 2 的 TTA 参数 ...
  --merge-cc-iou 0.08 \
  --merge-cc-dist 12 \
  --merge-cc-classes wall,door,window
```

**监控指标（每次必看）：**

| 指标 | 健康范围 | 异常处理 |
| --- | --- | --- |
| match_rate | 相对基线 **≥ +0.3pp** | 保留 |
| pred_gt_ratio | **0.95～1.10** | <0.90 说明过度合并，**立即废弃** |
| num_pred_instances | 不应比基线少 **>15%** | 对比 `b5_merge_wd`（2200 vs 3247） |

**验收**：相对 Step 2 最优，额外 **+0.5～1.5pp**；否则关闭 merge，仅保留 TTA。

---

#### 3.1.6 Step 4 — 面积过滤与匹配阈值（1 天）

**4a. Per-class min_area（需小改代码或扩展现有 CLI）**

当前 CLI 仅支持 `--min-area-shelf`；结构类碎片漏检可尝试：

| 方向 | 参数 | 风险 |
| --- | --- | --- |
| 降低全局 min_area | `--min-area 96` 或 `64` | pred 数上升，NMS 压力增大 |
| 降低 shelf | `--min-area-shelf 24` | 已部分支持，收益有限 |
| **扩展 CLI** | 为 wall/door 增加 `--min-area-wall` 等 | 需改 `build_extract_config()` |

**建议先做无改码 sweep：**

```bash
# 在 TTA 最优配置上
--min-area 96 --min-area-shelf 24
--min-area 112 --min-area-shelf 32   # 对照
```

**4b. Per-class IoU 匹配阈值**

CLI 已支持：`--iou-match-per-class door:0.22,wall:0.22,window:0.22`

```bash
OUT=outputs/match_improve/phase1_p1_iou_pc
# ... B1 或 TTA 基线参数 ...
  --iou-match 0.25 \
  --iou-match-per-class door:0.22,wall:0.22,window:0.22,shelf:0.22
```

**注意**：`b3_per_class_iou` 历史仅 +0pp；此项优先级低于 TTA/merge。  
**验收**：match +0.2pp 以上且 cls_on_matched 不下降 >0.5pp。

---

#### 3.1.7 Step 5 — 语义图轻量 refine（可选，1 天）

**代码**：`seg_refine.py`，在 CC **之前**作用于 label map。

**历史单因素均为负收益**，仅在 **TTA 已启用** 时小步尝试：

| 实验 ID | 参数 | 参考 |
| --- | --- | --- |
| p1_r1 | `--refine-morph-close 3 --refine-morph-classes wall,door,window` | 小 kernel，勿用 5 |
| p1_r2 | `--refine-dilate wall:1,door:1` | 勿超过 1px |
| p1_r3 | `--refine-split-door-wall` | adhesion 占比低，预期有限 |

**命令示例：**

```bash
  --seg-tta --seg-tta-scales 0.75,1.0,1.25 \
  --refine-morph-close 3 \
  --refine-morph-classes wall,door,window
```

**验收**：相对 TTA 基线 match **+0.2pp** 才纳入组合；否则跳过 Step 5。

---

#### 3.1.8 Step 6 — 组合验证与未匹配审计（1 天）

**6a. 全量组合跑（val 1000 张）：**

将 Step 2～5 中 **单项最优** 合并为一组参数，例如：

```bash
OUT=outputs/match_improve/phase1_best_candidate

python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --instance-source semantic \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --cls-checkpoint outputs/openclip_classifier/deliver_classifier_best.pth \
  --out-dir "${OUT}" \
  --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 \
  --iou-match 0.25 --min-area-shelf 32 \
  --seg-tta --seg-tta-scales 0.75,1.0,1.25
  # + 可选：--merge-cc-iou 0.08 --merge-cc-dist 12 --merge-cc-classes wall,door,window

python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${OUT}"
python transgrasp/pipelines/check_e1_gates.py --eval-dir "${OUT}"
```

**6b. 根因审计（必做）：**

```bash
python transgrasp/pipelines/export_unmatched_instances.py \
  --eval-dir outputs/match_improve/phase1_best_candidate \
  --out-dir outputs/match_improve/phase1_audit \
  --sample-wall 100 --sample-door 50 --render-vis
```

对比 `phase1_p0_b1_baseline` 的审计结果，确认 **miss** 占比是否下降（而非仅 iou_gap 口径变化）。

**6c. 产出台账：**

写入 `outputs/match_improve/phase1_summary.json`：

```json
{
  "baseline": "phase1_p0_b1_baseline",
  "best": "phase1_best_candidate",
  "baseline_match": 0.5916,
  "best_match": 0.0,
  "delta_pp": 0.0,
  "best_params": {},
  "per_class_delta": {"wall": 0, "door": 0}
}
```

---

#### 3.1.9 Step 7 — Phase 1 验收闸门

| 闸门 ID | 条件 | 说明 |
| --- | --- | --- |
| P1-PASS-1 | match_rate ≥ **63.0%** | 主目标（+3.8pp） |
| P1-PASS-2 | pred_gt_ratio ∈ **[0.95, 1.10]** | 防止 merge 过度 |
| P1-PASS-3 | wall_match ≥ **55%** | 结构类硬指标 |
| P1-PASS-4 | cls_on_matched 下降 ≤ **1.0pp** | 相对 B1 的 84.59% |
| P1-PASS-5 | strict_e2e ≥ **52%** | 系统级不倒退 |

**判定：**

- **全过**：更新 `run_deliver_semantic_e2e.sh` 与 `deliver_dual_track_manifest.json` 模式 A 指标。
- **仅过 P1-PASS-1/3 部分**：记录最优参数，进入 Phase 2；抓取仍用模式 B。
- **未过**：维持 B1 deploy；Phase 1 结论写入 `phase1_summary.json`，不强行上线。

**务实预期（基于已有实验）：**

| 手段 | 保守估计 | 乐观估计 |
| --- | --- | --- |
| TTA 单独 | +1.0～1.3pp | +1.5pp |
| 保守 merge | +0～0.8pp | +1.5pp |
| per-class iou/min_area | +0～0.3pp | +0.8pp |
| refine 组合 | +0～0.5pp | +1.0pp |
| **合计** | **~61～62%** | **~63～64%** |

达到 63% 可能需要 **TTA + 保守 merge + 少量 refine** 三者叠加；若仅 TTA 则约 **60.5%**，仍值得固化但不足以替代模式 B。

---

#### 3.1.10 建议新增脚本（Phase 1 工程化）

在 `scripts/run_segman_match_p1_sweep.sh` 中串联上述步骤（可参考 `scripts/run_e2e_improve_plan.sh`）：

```bash
#!/usr/bin/env bash
# Phase 1: B1 baseline → TTA → conservative merge → summarize
set -euo pipefail
cd "$(dirname "$0")/.."
IMPROVE=outputs/match_improve
BASE_ARGS=(--instance-source semantic
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth
  --cls-checkpoint outputs/openclip_classifier/deliver_classifier_best.pth
  --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10
  --iou-match 0.25 --min-area-shelf 32)

run_one() {
  local name="$1"; shift
  local out="${IMPROVE}/${name}"
  python transgrasp/pipelines/segment_and_classify.py \
    --eval-split val --max-images -1 --out-dir "${out}" \
    "${BASE_ARGS[@]}" "$@"
  python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${out}"
  python transgrasp/pipelines/check_e1_gates.py --eval-dir "${out}"
}

mkdir -p "${IMPROVE}"
run_one phase1_p0_b1_baseline
run_one phase1_p1_tta_default --seg-tta --seg-tta-scales 0.75,1.0,1.25
run_one phase1_p1_m2 --seg-tta --seg-tta-scales 0.75,1.0,1.25 \
  --merge-cc-iou 0.08 --merge-cc-dist 12 --merge-cc-classes wall,door,window
echo "Phase 1 sweep done -> ${IMPROVE}"
```

---

#### 3.1.11 Phase 1 时间线

| 天 | 任务 | 产出 | 状态 |
| --- | --- | --- | --- |
| D1 | Step 0～1 基线复现 | `phase1_p0_b1_baseline/` | ✅ 2026-06-08 |
| D2 | Step 2 TTA 复现 | `phase1_p1_tta_default` | ✅ 60.48% |
| D3～D4 | Step 3 merge 网格 | `phase1_p1_m1_*` / `phase1_p1_m2_*` | ✅ m1 无增益；m2 灾难 |
| D5 | Step 4 per-class iou | `phase1_p1_iou_per_class` | ✅ 60.84% 最优 |
| D6 | Step 5 可选 refine | — | ⏭ 跳过（历史单因素负收益） |
| D7～D8 | Step 6～7 审计 + 验收 | `phase1_summary.json` | ✅ P1_PASS=false |

**结题报告**：[Phase1_后处理优化实验报告.md](Phase1_后处理优化实验报告.md)

---

### Phase 2：弱类专项 finetune（数据 + 训练策略）

**总目标**：在 **不改 E2E 分类器** 的前提下，通过 **分割模型 finetune** 提升弱类 recall，将语义轨 match_rate 从 deploy 基线 **59.16%** 推到 **≥63%**（最低可接受），理想 **≥66%**。

**时间**：3～4 周（数据 1 周 + 训练 1.5 周 + 评测 0.5 周）。

**与 Phase 1 的分工**：

| 层 | Phase 1 结论 | Phase 2 应对 |
| --- | --- | --- |
| 后处理 / CC / merge | 天花板 **60.84%**，miss 仍 ~90% | **不再作为主手段** |
| 语义分割 recall | wall/door 漏检是主因 | **训练侧** copypaste + 难例重采样 |
| 分类 Acc | Phase 1 后处理未伤 cls | finetune 后必须复测 SegMAN-ROI |

**Phase 2 后处理口径**（全阶段统一，与 Phase 1 最优一致，便于叠加收益）：

```bash
--min-area 128 --nms-iou 0.5 --max-aspect-ratio 10
--iou-match 0.25 --min-area-shelf 32
--seg-tta --seg-tta-scales 0.75,1.0,1.25
--iou-match-per-class door:0.22,wall:0.22,window:0.22,shelf:0.22
```

---

#### 3.2.0 起点与量化目标

**起点（Phase 1 之后）**：

| 指标 | deploy B1 | Phase 1 后处理最优 | Phase 2 最低目标 | Phase 2 理想目标 |
| --- | --- | --- | --- | --- |
| match_rate | 59.16% | 60.84% | **≥63.0%** | **≥66.0%** |
| wall_match | 47.91% | 50.62% | **≥55%** | **≥58%** |
| door_match | 45.40% | 47.96% | **≥50%** | **≥53%** |
| pred_gt_ratio | 1.046 | 1.039 | [0.95, 1.10] | 同左 |
| mIoU（分割） | 81.80% | — | **≥81.0%** | **≥81.5%** |
| SegMAN-ROI Acc | 基准线 | — | 下降 **≤2pp** | 下降 **≤1pp** |
| strict_e2e | 50.05% | 51.63% | **≥53%** | **≥56%** |

**成功标准（分档）**：

- **P2-PASS（最低）**：E2E match ≥ 63%，wall ≥ 55%，ROI Acc 下降 ≤ 2pp → 可更新模式 A deploy。
- **P2-EXCELLENT**：E2E match ≥ 66%，wall ≥ 58% → 模式 A 可作为部分抓取场景的备选。
- **P2-FAIL**：match < 62% 或 ROI Acc 下降 > 3pp → 回退 `iter_6000.pth`，维持 B1 deploy。

---

#### 3.2.1 历史实验教训（Phase 2 必须遵守）

以下均来自 `outputs/e2e_improve/` 与 `docs/E2E端到端/E2E_实例匹配偏低根因与改进方案.md`：

| 实验 | 手段 | 结果 | Phase 2 处置 |
| --- | --- | --- | --- |
| **P0 弱类 finetune** | 仅提高 class_weight | SegMAN-ROI Acc **61.81%**（大幅下降） | ❌ **禁止** 单独加 loss weight |
| **E2-1 e2weak** | boundary loss 0.22 + 弱类 schedule | match **58.39%**（低于 B1） | ❌ boundary alone 无效 |
| **e2copypaste（Scheme C）** | copypaste + class_weight | 代码已有，**E2E 未系统验收** | ✅ **Phase 2 主方案** |
| Phase 1 merge | merge-cc-iou ≥ 0.08 | match **43.67%** | ❌ 训练后亦禁止高 IoU merge |

**Phase 2 原则**：

1. **数据增强优先**：Copy-Paste 为主、class_weight 为辅（幅度 ≤ P0/e2copypaste 水平）。
2. **低 LR + 短 schedule**：热启动 `iter_6000.pth`，总 iter ≤ 6000，避免过拟合伤 ROI。
3. **双指标早停**：同时看 **mIoU** 与 **proxy match_rate**，不以 mIoU alone 选 ckpt。
4. **两阶段解冻**：先 decoder、后全模型，保护已收敛的 backbone 特征。
5. **评测叠加 Phase 1 后处理**：新 ckpt 的收益应在 TTA + per-class IoU 口径下衡量。

---

#### 3.2.2 代码与配置清单（已有 / 待建）

| 类型 | 路径 | 状态 |
| --- | --- | --- |
| Patch bank 构建 | `segmentation/tools/build_copypaste_patch_bank.py` | ✅ 已有 |
| Copy-Paste pipeline | `segmentation/mmseg/datasets/pipelines/trans10k_copypaste.py` | ✅ 已有 |
| E2 copypaste config（参考） | `segman_b_trans10k_lass_balanced_v2_e2copypaste.py` | ✅ 参考基线 |
| E2 weak config（反面教材） | `segman_b_trans10k_lass_balanced_v2_e2weak.py` | ⚠️ 勿单独使用 |
| P0 weak config（反面教材） | `segman_b_trans10k_lass_balanced_v2_p0weak.py` | ⚠️ 勿单独使用 |
| E2 训练脚本（参考） | `scripts/run_e2_copypaste_train.sh` | ✅ 可改作 P2 |
| **Phase 2 专用 config** | `segman_b_trans10k_lass_match_p2.py` | 🔲 待新建 |
| **Phase 2 训练脚本** | `scripts/run_segman_match_p2_finetune.sh` | 🔲 待新建 |
| **Proxy match 工具** | `transgrasp/pipelines/eval_proxy_match.py` | 🔲 待新建 |
| 难例 stem 列表 | `outputs/match_improve/phase1_audit_best/candidate_stems_wall_door.txt` | ✅ Phase 1 产出 |

---

#### 3.2.3 Step 0 — 环境与基线锁定（0.5 天）

**目的**：确保 Phase 2 所有实验相对同一基线，E2E 评测口径与 Phase 1 一致。

**目标**：环境就绪；B1 + Phase 1 最优后处理指标可复现。

```bash
cd /workspace/segman   # Docker segman_train
source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman

# 1) 分割权重
test -f segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth

# 2) 分类器（全程不变）
test -f outputs/openclip_classifier/deliver_classifier_best.pth

# 3) Phase 1 台账
cat outputs/match_improve/phase1_summary.json

# 4) 数据
test -d segmentation/data/trans10k/img_dir/train
test -d segmentation/data/trans10k/ann_dir/train
```

**验收**：

| 检查项 | 期望 |
| --- | --- |
| `phase1_p0_b1_baseline` match | 59.16% ± 0.2pp |
| `phase1_p1_iou_per_class` match | 60.84% ± 0.2pp |
| `iter_6000.pth` mIoU | 81.80% ± 0.2pp |

---

#### 3.2.4 Step 1 — 数据准备：Patch Bank + 难例列表（2～3 天）

**目的**：针对 **miss 主导**（~90%）的漏检，在训练侧增加弱类实例 exposure，而非依赖后处理。

**目标**：生成可用 patch bank；难例 stem 列表就绪；copypaste 可视化抽检通过。

##### 1a. 构建弱类 Patch Bank

```bash
cd segmentation

python tools/build_copypaste_patch_bank.py \
  --data-root data/trans10k \
  --split train \
  --out data/trans10k/copypaste_patch_bank_p2.pkl \
  --paste-classes 7,9,3,11 \
  --min-area 64 --max-area 8192 \
  --max-patches-per-class 800
```

| 参数 | 值 | 说明 |
| --- | --- | --- |
| paste-classes | `7,9,3,11` | door, wall, window, shelf（Phase 2 核心弱类） |
| max-patches-per-class | 800 | 比 E2 默认 400 加倍，强化 exposure |

**验收**：`copypaste_patch_bank_p2.pkl` 存在；每类 patch 数 ≥ 200。

##### 1b. 难例 stem 列表（训练过采样）

复用 Phase 1 审计产出，并扩展 train 侧难例：

```bash
# 已有：val 难例（wall/door 未匹配）
cp outputs/match_improve/phase1_audit_best/candidate_stems_wall_door.txt \
   outputs/match_improve/phase2_hard_stems_val.txt

# 可选：从 train 标注统计低 recall 场景（脚本待写或手工整理）
# 输出：outputs/match_improve/phase2_hard_stems_train.txt
```

**目的**：训练时对这些 stem **×2～3 过采样**，与 copypaste 互补。

##### 1c. Copy-Paste 冒烟可视化

```bash
cd segmentation
python tools/smoke_copypaste.py \
  --config local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2_e2copypaste.py \
  --num-samples 20 --out-dir ../outputs/match_improve/phase2_copypaste_vis
```

**验收**：粘贴后 label 无大面积空洞；door/wall/window/shelf patch 可见且类 ID 正确。

---

#### 3.2.5 Step 2 — 新建 Phase 2 训练 Config（1 天）

**目的**：在 `balanced_v2` 架构上，组合 **copypaste + 温和 class_weight + 两阶段 LR**，避免 P0/E2-1 失败模式。

**目标**：产出 `segman_b_trans10k_lass_match_p2.py`，含 Stage-1 / Stage-2 两套 `cfg-options` 约定。

**建议新建**：`segmentation/local_configs/segman_trans/segman_b_trans10k_lass_match_p2.py`

| 配置块 | Stage-1（decoder only） | Stage-2（full finetune） |
| --- | --- | --- |
| `load_from` | `iter_6000.pth` | Stage-1 best ckpt |
| 冻结 | `backbone` 全冻结 | 全部解冻 |
| `optimizer.lr` | **5e-5**（head ×10） | **1e-5** |
| `max_iters` | **2000** | **4000** |
| `boundary_loss_weight` | 0.18（保持 v2） | **0.22**（略升，勿超 0.25） |
| Copy-Paste | `paste_prob=0.5, max_paste=2` | 同左 |
| class_weight | door×1.20, wall×0.95, window×1.15, shelf×1.25 | 同左 |
| Dice loss | 0.15 | 0.15 |
| 早停监控 | 每 500 iter 存 ckpt | 每 500 iter 存 ckpt |

**class_weight 原则**（相对 `balanced_v2`）：

- wall 保持 **≤1.0**（防吞噬 door/window）
- door/window/shelf **适度** 提升（1.15～1.25），禁止 P0 式全面拉高
- cup/bowl/eyeglass **不变**，保护已达标小物体

**train_pipeline 关键段**（在 `RandomCrop` 之前插入）：

```python
dict(
    type='Trans10KCopyPaste',
    patch_bank='data/trans10k/copypaste_patch_bank_p2.pkl',
    paste_prob=0.5,
    max_paste=2,
    paste_classes=[7, 9, 3, 11],  # door, wall, window, shelf
    scale_range=(0.8, 1.2),
    max_overlap=0.3,
),
```

**验收**：`python tools/train.py <cfg> --work-dir /tmp/p2_smoke --cfg-options runner.max_iters=10` 可跑通 10 iter。

---

#### 3.2.6 Step 3 — Stage-1 训练：冻结 Backbone（2～3 天）

**目的**：在 **不破坏 backbone 通用特征** 的前提下，先让 decoder（MMSCopE）适应弱类 recall 导向的增广。

**目标**：Stage-1 结束时有 2～4 个候选 ckpt；proxy match 相对 B1 **≥ +1.0pp**。

```bash
cd segmentation
WORK=outputs/trans10k_lass_match_p2_stage1
CFG=local_configs/segman_trans/segman_b_trans10k_lass_match_p2.py

python tools/train.py "${CFG}" \
  --work-dir "${WORK}" \
  --cfg-options \
    runner.max_iters=2000 \
    optimizer.lr=5e-5 \
    model.backbone.frozen_stages=4 \
    load_from=outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth

# 分割 mIoU
for it in 500 1000 1500 2000; do
  python tools/test.py "${CFG}" --checkpoint "${WORK}/iter_${it}.pth" \
    --eval mIoU --work-dir "${WORK}/eval_iter_${it}"
done
```

**监控指标（每 500 iter）**：

| 指标 | 健康范围 | 异常处理 |
| --- | --- | --- |
| train loss | 平稳下降 | 震荡 → LR 减半 |
| val mIoU | ≥ 80.5% | < 80% → 停止 Stage-1 |
| proxy match（Step 5） | ≥ B1 + 1pp | 无提升 → 调 paste_prob / 难例采样 |

**验收**：`iter_1000` 或 `iter_2000` 的 proxy match ≥ **60.2%**（B1 59.16% + 1pp）。

---

#### 3.2.7 Step 4 — Stage-2 训练：全模型低 LR Finetune（3～5 天）

**目的**：在 Stage-1 基础上微调全网络，进一步抬升 wall/door 边界与 recall。

**目标**：产出最终候选 ckpt；proxy match ≥ **62%**（距 P2 最低目标 63% 留 1pp 给全量 E2E）。

```bash
STAGE1_BEST="${WORK}/iter_2000.pth"   # 或 proxy 最优 ckpt
WORK2=outputs/trans10k_lass_match_p2_stage2

python tools/train.py "${CFG}" \
  --work-dir "${WORK2}" \
  --cfg-options \
    runner.max_iters=4000 \
    optimizer.lr=1e-5 \
    model.decode_head.mmscope_cfg.boundary_loss_weight=0.22 \
    load_from="${STAGE1_BEST}"
```

**实验网格（可选 ablation，子集 100 张加速）**：

| 实验 ID | boundary | paste_prob | max_iters | 说明 |
| --- | --- | --- | --- | --- |
| p2_s2_a | 0.20 | 0.5 | 3000 | 保守 |
| p2_s2_b | 0.22 | 0.5 | 4000 | **默认** |
| p2_s2_c | 0.22 | 0.6 | 4000 | 加强 copypaste |
| p2_s2_d | 0.25 | 0.5 | 4000 | 边界加强（警惕 E2-1 回退） |

**验收**：至少 1 个 ckpt 满足 proxy match ≥ **62%** 且 mIoU ≥ **81.0%**。

---

#### 3.2.8 Step 5 — Proxy Match 快速筛选（贯穿 Step 3～4）

**目的**：全量 E2E（1000 张 × TTA × 分类）耗时数小时；训练期用 **固定后处理、不跑分类** 的 proxy 指标快速筛 ckpt。

**目标**：实现并固化 `eval_proxy_match.py`；100 张 val 子集 **< 15 min/ckpt**。

**待新建**：`transgrasp/pipelines/eval_proxy_match.py`

**逻辑**：

```text
val 子集（默认 100 张）
  → SegMANSegmentor.predict_label_map()   # 无 TTA（训练期加速）
  → extract_instance_rois + postprocess_instances
  → match_instances_to_gt（iou 0.25）
  → 输出 proxy_match_rate + per_class wall/door
```

**固定参数**（与 B1 一致，训练期不用 TTA）：

```bash
python transgrasp/pipelines/eval_proxy_match.py \
  --seg-checkpoint segmentation/outputs/trans10k_lass_match_p2_stage2/iter_4000.pth \
  --eval-split val --max-images 100 \
  --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 \
  --iou-match 0.25 --min-area-shelf 32 \
  --out outputs/match_improve/phase2_proxy/iter_4000.json
```

**proxy 与全量 E2E 换算（经验）**：

| proxy（无 TTA, 100 张） | 预期全量 E2E（+TTA+per-class IoU） |
| --- | --- |
| 60% | ~61.5～62.0% |
| 62% | ~63.5～64.0% |
| 64% | ~65.5～66.0% |

**验收**：工具可运行；对 `iter_6000.pth` proxy 与 B1 全量 match 偏差 < 1pp。

---

#### 3.2.9 Step 6 — 全量 E2E 评测（1～2 天）

**目的**：对 proxy 最优的 1～3 个 ckpt 做 **正式 val 1000 张** E2E，叠加 Phase 1 最优后处理。

**目标**：选出 `phase2_best_candidate`；match ≥ 63% 或确认失败。

```bash
CKPT=segmentation/outputs/trans10k_lass_match_p2_stage2/iter_XXXX.pth
OUT=outputs/match_improve/phase2_best_candidate

python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --instance-source semantic \
  --seg-checkpoint "${CKPT}" \
  --cls-checkpoint outputs/openclip_classifier/deliver_classifier_best.pth \
  --out-dir "${OUT}" \
  --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 \
  --iou-match 0.25 --min-area-shelf 32 \
  --seg-tta --seg-tta-scales 0.75,1.0,1.25 \
  --iou-match-per-class door:0.22,wall:0.22,window:0.22,shelf:0.22

python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${OUT}"
python transgrasp/pipelines/export_unmatched_instances.py \
  --eval-dir "${OUT}" \
  --out-dir outputs/match_improve/phase2_audit \
  --sample-wall 100 --sample-door 50 --render-vis
```

**对比基线**：

| 对比项 | 目录 |
| --- | --- |
| deploy B1 | `outputs/match_improve/phase1_p0_b1_baseline` |
| Phase 1 后处理最优 | `outputs/match_improve/phase1_p1_iou_per_class` |
| Phase 2 候选 | `outputs/match_improve/phase2_best_candidate` |

**验收**：`e2e_metrics_report.json` 生成；相对 B1 match **≥ +3.8pp** 为 P2-PASS 最低线。

---

#### 3.2.10 Step 7 — SegMAN-ROI 分类 Acc 复测（0.5 天）

**目的**：finetune 可能改变语义 mask 分布，导致 **同一分类器** 在 SegMAN-ROI 上 Acc 下降（P0 教训）。

**目标**：确认 ROI Acc 下降 ≤ 2pp；否则不得替换 deploy 分割权重。

```bash
# 用新 ckpt 重新导出 SegMAN-ROI（若已有流水线）
# 或复用 deliver_p3 评测脚本，仅替换分割来源

python transgrasp/classification/eval_roi_classifier.py \
  --checkpoint outputs/openclip_classifier/deliver_classifier_best.pth \
  --roi-root data/trans10k_roi_segman_p2 \
  --out-dir outputs/match_improve/phase2_roi_eval
```

**注意**：若新 mask 未导出 ROI，可先用 **E2E cls_on_matched** 作代理（Phase 1 已证与 ROI 趋势一致），但正式验收仍需 ROI 评测。

| 指标 | 基准 | 阈值 |
| --- | --- | --- |
| SegMAN-ROI top-1 Acc | deliver 基准 | 下降 ≤ **2.0pp** |
| E2E cls_on_matched | 84.59%（B1） | 下降 ≤ **1.0pp** |

---

#### 3.2.11 Step 8 — 验收闸门与 Deploy 决策（0.5 天）

**目的**：按量化闸门决定是否替换模式 A 分割权重、是否进入 Phase 3。

| 闸门 ID | 条件 | 说明 |
| --- | --- | --- |
| P2-PASS-1 | E2E match ≥ **63.0%** | 相对 B1 +3.8pp（弥补 Phase 1 欠账） |
| P2-PASS-2 | E2E match ≥ **66.0%** | 理想目标 |
| P2-PASS-3 | wall_match ≥ **55%** | 结构类硬指标 |
| P2-PASS-4 | pred_gt_ratio ∈ [0.95, 1.10] | 防止 recall 提升伴随过度过分割 |
| P2-PASS-5 | mIoU ≥ **81.0%** | 分割质量底线 |
| P2-PASS-6 | SegMAN-ROI Acc 下降 ≤ **2pp** | P0 教训 |
| P2-PASS-7 | cls_on_matched 下降 ≤ **1pp** | 分类耦合 |
| P2-PASS-8 | strict_e2e ≥ **53%** | 系统级不倒退 |

**判定**：

| 结果 | 动作 |
| --- | --- |
| P2-PASS-1 + 3 + 6 全过 | 更新 `run_deliver_semantic_e2e.sh`、`deliver_dual_track_manifest.json`；写入 Phase 2 报告 |
| 仅过 P2-PASS-1 | 记录最优 ckpt 与参数，模式 A 可选升级，抓取仍用 B |
| 未过 | **回退 `iter_6000.pth`**；Phase 2 结论写入 `phase2_summary.json`，不进入 Phase 3 |

**产出台账**：`outputs/match_improve/phase2_summary.json`

```json
{
  "baseline_seg": "iter_6000.pth",
  "best_ckpt": "iter_XXXX.pth",
  "baseline_match_b1": 0.5916,
  "phase1_best_match": 0.6084,
  "phase2_e2e_match": 0.0,
  "delta_pp_vs_b1": 0.0,
  "roi_acc_delta_pp": 0.0,
  "P2_PASS": false,
  "deploy_action": "rollback | promote"
}
```

---

#### 3.2.12 建议脚本 `run_segman_match_p2_finetune.sh`

```bash
#!/usr/bin/env bash
# Phase 2: patch bank → stage1 → stage2 → proxy screen → E2E eval
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segman

IMPROVE=outputs/match_improve
CFG=segmentation/local_configs/segman_trans/segman_b_trans10k_lass_match_p2.py

echo "=== P2 Step 1: Patch bank ==="
cd segmentation
python tools/build_copypaste_patch_bank.py \
  --out data/trans10k/copypaste_patch_bank_p2.pkl \
  --paste-classes 7,9,3,11 --max-patches-per-class 800

echo "=== P2 Step 3: Stage-1 ==="
python tools/train.py "${CFG}" \
  --work-dir outputs/trans10k_lass_match_p2_stage1 \
  --cfg-options runner.max_iters=2000 optimizer.lr=5e-5 \
    model.backbone.frozen_stages=4

echo "=== P2 Step 4: Stage-2 ==="
S1_CKPT=outputs/trans10k_lass_match_p2_stage1/iter_2000.pth
python tools/train.py "${CFG}" \
  --work-dir outputs/trans10k_lass_match_p2_stage2 \
  --cfg-options runner.max_iters=4000 optimizer.lr=1e-5 \
    load_from="${S1_CKPT}" \
    model.decode_head.mmscope_cfg.boundary_loss_weight=0.22

echo "=== P2 Step 5-6: Proxy + E2E (manual best ckpt selection) ==="
echo "See docs/优化SegMANmatch_rate/SegMAN_match_rate_提升实施计划.md §3.2.8-3.2.9"
```

---

#### 3.2.13 Phase 2 时间线

| 天 | 任务 | 产出 | 里程碑 |
| --- | --- | --- | --- |
| D1 | Step 0 基线锁定 | 环境检查记录 | 基线可复现 |
| D2～D4 | Step 1 数据准备 | `copypaste_patch_bank_p2.pkl`、难例 stem | 数据 ready |
| D5 | Step 2 新建 config | `segman_b_trans10k_lass_match_p2.py` | config 可训练 |
| D6～D8 | Step 3 Stage-1 | `match_p2_stage1/iter_*.pth` | proxy ≥ 60.2% |
| D9～D14 | Step 4 Stage-2 + ablation | `match_p2_stage2/iter_*.pth` | proxy ≥ 62% |
| D15 | Step 5 proxy 工具 + 筛 ckpt | `phase2_proxy/*.json` | 选定 1～3 候选 |
| D16～D17 | Step 6 全量 E2E | `phase2_best_candidate/` | match 数值 |
| D18 | Step 7 ROI Acc | `phase2_roi_eval/` | Acc 下降 ≤ 2pp |
| D19～D20 | Step 8 闸门 + 报告 | `phase2_summary.json`、结题报告 | Go / No-Go |

---

#### 3.2.14 风险与回退

| 风险 | 概率 | 缓解 |
| --- | --- | --- |
| Copy-Paste 实现边界 artifact | 中 | `smoke_copypaste.py` 可视化 20 张 |
| 训练后 mIoU 降 > 1pp | 中 | 低 LR + 早停；mIoU 闸门 81% |
| ROI Acc 大幅下降（P0 重演） | 中 | 禁止单独加 loss weight；Stage-1 冻结 backbone |
| proxy 与全量 E2E 偏差大 | 低 | 最终仍以 Step 6 全量为准 |
| 算力不足 | 低 | 先跑 100 张 proxy + 50 张 E2E 子集 |

**回退**：任何阶段 FAIL → 保留 `iter_6000.pth` 为模式 A 分割权重；抓取轨始终可用模式 B（75.46%）。

---

#### 3.2.15 务实预期

| 手段 | 保守估计 | 乐观估计 |
| --- | --- | --- |
| Copy-Paste alone | +1.5～2.5pp | +3～4pp |
| + 难例过采样 | +0.5～1pp | +1～2pp |
| + 两阶段 finetune | +0.5～1pp | +1～2pp |
| + Phase 1 后处理（TTA+IoU） | +1.5pp（已验证） | +1.7pp |
| **合计（相对 B1）** | **~63～65%** | **~66～68%** |

达到 **66%** 需要训练与后处理 **同时** 有效；若训练仅 +2pp，叠加 Phase 1 后处理仍可能达到 **~63%**（P2 最低可接受线）。

---

### Phase 3：架构级探索（可选，成本高）

仅在 Phase 2 未达标且业务强烈要求「纯 SegMAN 轨」时考虑。

#### 3.3.1 轻量实例头
- 在 SegMAN encoder 后添加 Query-based instance head（参考 Mask2Former）。
- 训练目标：semantic + instance 联合 loss。

#### 3.3.2 Hybrid 融合
- wall/door/window/shelf 使用 M2F 实例，其余类仍用 SegMAN CC。
- 需要修改 `instance_predictor.py` 支持多源实例融合。

**风险**：偏离当前双轨交付定位，建议作为「未来工作」而非主线。

---

## 4. 实验管理与工具

### 4.1 目录规范
```
outputs/match_improve/
├── phase1_cc_merge/          # Phase 1 参数 sweep
├── phase2_finetune/          # 新 checkpoint + proxy match 报告
├── phase2_e2e_eval/          # 全量 E2E 评测
└── ablation_logs/            # 每次实验的 config + log + metrics
```

### 4.2 自动化工具（需新增或复用）
- `scripts/run_segman_match_p1_sweep.sh`：Phase 1 参数网格搜索。
- `transgrasp/pipelines/eval_proxy_match.py`：快速 proxy match_rate 计算（不跑完整分类）。
- `viz_deliver/` 扩展：自动生成 match_rate 提升对比图。

### 4.3 版本控制
- 每次实验 commit message 格式：
  ```
  exp: match_p1_cc_merge_iou0.15_dist12 [match=0.632]
  ```

---

## 5. 时间线与资源

| 阶段 | 起止 | 负责人 | 交付物 | 里程碑 |
|------|------|--------|--------|--------|
| Phase 1 | Week 1~2 | 算法工程师 | 参数 sweep 报告 + 新 E2E 指标 | match ≥ 63% |
| Phase 2 数据准备 | Week 2~3 | 数据工程师 | copypaste patch bank + 采样权重 | 数据 ready |
| Phase 2 训练 | Week 3~5 | 算法工程师 | 新 checkpoint + proxy match 报告 | proxy ≥ 66% |
| Phase 2 验收 | Week 5~6 | 全员 | 全量 E2E 评测 + 分类 Acc 报告 | E2E match ≥ 66% |
| 决策 Gate | Week 6 | 项目组 | 是否进入 Phase 3 | Go / No-Go |

**总周期**：6 周（含缓冲）。

---

## 6. 风险与回退

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| Phase 1 增益 < 3pp | 中 | 延误 Phase 2 | 提前并行准备 copypaste 数据 |
| Phase 2 SegMAN-ROI Acc 大幅下降 | 中 | 分类模块失效 | 早停 + 混合验证集 |
| 训练不稳定（loss 震荡） | 低 | 浪费算力 | 使用更低 LR + gradient clipping |
| Phase 3 成本过高 | 高 | 项目延期 | 决策 Gate 时明确 No-Go 条件 |

**回退策略**：
- 若 Phase 1 失败，直接采纳模式 B 作为唯一抓取轨。
- 若 Phase 2 分类 Acc 下降 > 3pp，保留 Phase 1 成果，放弃 finetune。

---

## 7. 预期成果与交付

### 7.1 量化成果
- **Phase 1**：match_rate **63~65%**，wall_match **≥ 55%**。
- **Phase 2**：match_rate **66~68%**，SegMAN-ROI 分类 Acc 下降 ≤ 2pp。
- **可选 Phase 3**：match_rate **≥ 70%**（仅在业务需要时）。

### 7.2 文档与代码交付
- 更新 `docs/交付/SegMAN_OpenCLIP_E2E_交付路线.md`（若 Phase 2 成功）。
- 新增实验报告：`outputs/match_improve/phase2_final_report.md`。
- 脚本：`scripts/run_segman_match_p{1,2}_*.sh`。
- 可视化：扩展 `viz_deliver/` 生成 match_rate 提升对比图。

### 7.3 对双轨交付的影响
- 若 Phase 2 成功，可在 `deliver_dual_track_manifest.json` 中更新模式 A 的 match_rate 指标。
- 否则维持现状，明确模式 A 仅用于语义场景，抓取默认模式 B。

---

## 8. 附录：快速启动命令

```bash
# Phase 1 全流程 sweep（已提供脚本）
bash scripts/run_segman_match_p1_sweep.sh

# 子集快速验证（50 张）
bash scripts/run_segman_match_p1_sweep.sh --quick

# Phase 2 全流程（详见 §3.2.12）
docker exec segman_train bash -lc '
  source /root/anaconda3/etc/profile.d/conda.sh &&
  conda activate segman &&
  cd /workspace/segman &&
  bash scripts/run_segman_match_p2_finetune.sh
'

# Phase 2 仅数据准备（Step 1）
cd segmentation && python tools/build_copypaste_patch_bank.py \
  --out data/trans10k/copypaste_patch_bank_p2.pkl \
  --paste-classes 7,9,3,11 --max-patches-per-class 800
```

---

**本计划为 SegMAN 语义轨 match_rate 提升的完整路线图，建议项目组在 Week 1 启动 Phase 1 参数 sweep，并于 Week 6 进行 Go/No-Go 决策。**