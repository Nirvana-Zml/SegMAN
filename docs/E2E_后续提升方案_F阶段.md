# E2E 实例匹配 — 后续提升方案（F 阶段）


| 项目   | 内容                                      |
| ---- | --------------------------------------- |
| 文档版本 | v1.2                                    |
| 编写日期 | 2026-05-27                              |
| 前置文档 | 《E2E_实例匹配偏低根因与改进方案.md》（方案 B～E 已全部执行）    |
| 评测基准 | val 全量 1000 张 / **3105** GT 实例；B1 后处理口径 |
| 当前基线 | **v2@6k + B1**，match **59.16%**         |


---

## 1. 执行摘要

方案 **B → E** 已在相同 E2E 口径下完成系统实验。**后处理、短训、推理 refine、Mask R-CNN MVP** 均无法将 match 稳定抬过 **61%**。

**GT oracle 实验（92.98% match）** 证明：分类栈与匹配逻辑可用，**瓶颈在「语义/实例 mask 对 GT 实例的 recall」**，尤其 **wall/door miss**。

F 阶段目标：**继续提升实例 match**，主攻 **F1 实例分割长训**；F1 不足时再启动 **F2 结构性突破**。避免重复 B5/D morph/C Copy-Paste/MRCNN MVP 等已证无效方向。

---

## 2. 方案 B～E 实验台账

### 2.1 全量结果一览


| 方案                | 措施                | match_rate | pred/GT   | strict E2E | 结论         |
| ----------------- | ----------------- | ---------- | --------- | ---------- | ---------- |
| baseline          | v2@6k 语义，无 E1     | 59.32%     | 1.148     | ≈49.9%     | 冗余偏高       |
| **B1**            | min_area128 + NMS | **59.16%** | **1.046** | **50.05%** | **当前基线**   |
| B2 匈牙利            | 匹配算法              | 59.16%     | 1.046     | 50.18%     | 无提升        |
| B5 CC 合并          | merge_iou=0.3     | 44.99%     | 0.709     | 38.74%     | ❌ 灾难       |
| C Copy-Paste      | 分割再训练             | 58.62%     | 1.101     | 49.66%     | ❌ FAIL     |
| D morph/dilate    | 推理 refine         | 58.2～58.4% | 1.03～1.09 | ≈49.4%     | ❌ 有害       |
| D4 TTA            | 多尺度 flip          | 60.48%     | 1.039     | 51.37%     | 参考（×6 耗时）  |
| **E2 GT oracle**  | GT mask 上界        | **92.98%** | 0.930     | 73.37%     | **改进空间上界** |
| E4 Mask R-CNN MVP | 1500×5ep          | 29.28%     | 0.660     | 25.60%     | ❌ 欠训       |


### 2.2 根因


| 根因              | 占比             | 说明                       |
| --------------- | -------------- | ------------------------ |
| **miss 漏检**     | **~82%**       | pred 与 GT max IoU < 0.10 |
| iou_gap         | ~18%           | 有重叠但 IoU < 0.25          |
| wall + door 未匹配 | **81%** 占全部未匹配 | 结构类 recall 不足            |


### 2.3 已排除方向（勿重复）

- 同类 CC 粗合并（B5）
- 语义 mask morph/dilate（D1/D2）
- Copy-Paste 短训 finetune（C）
- 短训 Mask R-CNN MVP（E4）
- 仅调 IoU 阈值 / 匈牙利匹配（B2/B3）

---

## 3. F 阶段路线总览

```text
                    ┌─────────────────────────────────────┐
  F1 模型提升       │ Mask2Former 全量长训（E1 完整版）      │  2～4 周  ← 当前
                    └─────────────────────────────────────┘
                              ↓ F1 未 PASS
                    ┌─────────────────────────────────────┐
  F2 结构性突破     │ 标注升级 / wall-door 专项 / 混合架构  │  1～2 月
                    └─────────────────────────────────────┘
```


| 路线     | 目标 match   | 工期    | 算力        | 说明         |
| ------ | ---------- | ----- | --------- | ---------- |
| **F1** | **65～72%** | 2～4 周 | 1×GPU·2 周 | **当前主攻**   |
| F2     | **75%+**   | 1～2 月 | 高         | F1 不足或冲击上限 |


---

## 4. F1 — Mask2Former 全量长训（2～4 周）

**定位**：完成方案 E 未完成的 **E1 完整版**；MVP Mask R-CNN（29%）**不代表**实例分割上限。


| 编号   | 步骤                        | 工期      | 说明                         |
| ---- | ------------------------- | ------- | -------------------------- |
| F1-0 | 确认基线 + COCO 伪实例           | 0.5 天   | 对比口径固定                     |
| F1-1 | 环境安装（mmdet / Mask2Former） | 0.5～1 天 | Docker 内                   |
| F1-2 | 编写训练 config               | 0.5 天   | 待开发                        |
| F1-3 | 全量训练 40k iter             | 1～2 周   | 主耗时                        |
| F1-4 | val 推理 + 分割级 match        | 0.5 天   | 早停依据                       |
| F1-5 | E2E 接入（m2f 后端）            | 1～2 天   | 待开发                        |
| F1-6 | 全量 E2E + E2-0 审计          | 0.5～1 天 | F1-PASS 判定                 |
| F1-7 | 未 PASS → 调参二轮             | 3～5 天   | num_queries / class_weight |


### 4.1 F1 执行手册（步骤 / 命令 / 目的）

> **工期**：2～4 周（环境 1 天 + config 0.5 天 + 训练 1～2 周 + 接入/验收 2 天）。  
> **前置**：方案 E **E0 伪实例 COCO** 已导出；B1 基线 match **59.16%**；GT oracle **92.98%** 已证上界。  
> **环境**：Docker `segman_train`，`/workspace/segman`；`conda activate segman`。  
> **评测口径**：E2E 仍用 B1 `--min-area 128 --nms-iou 0.5 --iou-match 0.25 --min-area-shelf 32`；**pred 实例来自 Mask2Former**，不再语义 CC。

---

##### 总流程

```text
Step 0  F1-0  确认 B1 基线 + COCO 伪实例台账
  ↓
Step 1  F1-1  安装 mmdet / mmengine / Mask2Former 配置
  ↓
Step 2  F1-2  编写 m2f_trans10k_pseudo_instances.py config
  ↓
Step 3  F1-3  全量 train 40k iter（COCO 预训练 init）
  ↓
Step 4  F1-4  每 5k iter：val 推理 → 分割级 instance match
  ↓
Step 5  F1-5  实现 m2f 后端 + segment_and_classify CLI
  ↓
Step 6  F1-6  最优 ckpt 全量 E2E + E2-0 审计
  ↓
Step 7  F1-PASS → 更新实例源；FAIL → F1-7 调参或进入 F2
```

**验收汇总表**（相对 B1 `match≈59.16%`）：


| 指标            | B1     | **F1 目标**             | E2 oracle |
| ------------- | ------ | --------------------- | --------- |
| match_rate    | 59.16% | **≥65%**（stretch 70%） | 92.98%    |
| pred_gt_ratio | 1.046  | **0.95～1.08**         | 0.930     |
| strict E2E    | 50.05% | **≥55%**              | 73.37%    |
| wall match    | 46.4%  | **≥50%**（stretch 55%） | 91.6%     |
| door match    | 45.9%  | **≥50%**              | —         |
| miss 占比（审计）   | 82%    | **≤65%**              | —         |


**F1-PASS 闸门**：


| 闸门        | 条件                                      |
| --------- | --------------------------------------- |
| F1-PASS-1 | match_rate **≥65%**                     |
| F1-PASS-2 | pred_gt_ratio **∈ [0.95, 1.08]**        |
| F1-PASS-3 | strict E2E **≥55%**                     |
| F1-PASS-4 | `e2e_top1_on_matched` **≥83%**          |
| F1-PASS-5 | wall match **≥50%** 或 wall 未匹配 **≤500** |
| F1-FAIL   | match **<62%** → **保留 B1**，进入 F2        |


**分割级早停闸门**（F1-4，无分类，用于训练中期决策）：


| 闸门          | 条件                         | 动作         |
| ----------- | -------------------------- | ---------- |
| F1-SEG-1    | 实例 match **≥62%**          | 继续训至 40k   |
| F1-SEG-2    | wall/door match 各 **≥45%** | 继续         |
| F1-SEG-FAIL | iter 20k 后 match **<55%**  | 停训调 config |


---

##### F1-0 — 确认基线与 COCO 伪实例（0.5 天）

**目的**：固定对比口径；确认 E0 数据可用于 mmdet。

```bash
cd /workspace/segman
source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman

cat outputs/e2e_improve/b_plan_summary.json
cat outputs/e2e_improve/e_plan_summary.json

# 若无 COCO 则重新导出
python segmentation/tools/export_trans10k_coco_instances.py \
  --data-root segmentation/data/trans10k \
  --splits train,val \
  --min-area 64 \
  --out-dir segmentation/data/trans10k/coco_instances

# 目视抽查
python segmentation/tools/browse_coco_instances.py \
  --ann segmentation/data/trans10k/coco_instances/val.json \
  --img-dir segmentation/data/trans10k/img_dir/val \
  --max-images 20 \
  --out-dir outputs/e2e_improve/f1_coco_browse

# B1 基线复现
OUT=outputs/e2e_improve/f1_b1_ref \
bash scripts/run_e2e_regression.sh
```

**验收**：


| 项                 | 标准                                  |
| ----------------- | ----------------------------------- |
| val 伪实例数          | **3105**（±1%）                       |
| train 伪实例数        | **15746**                           |
| `f1_b1_ref` match | **≈0.5916**，`num_gt_instances=3105` |


---

##### F1-1 — 环境安装（0.5～1 天）

**目的**：在 Docker 内安装 mmdet 3.x 与 Mask2Former 依赖，与现有 mmseg 共存。

```bash
conda activate segman
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

# 验证
python -c "import mmdet; print('mmdet', mmdet.__version__)"
python -c "import mmengine; print('mmengine', mmengine.__version__)"

# 下载 Mask2Former 配置（任选其一）
# 方式 A：从 mmdet model zoo 复制 config 到 local_configs/mask2former/
# 方式 B：git clone mmdetection/configs/mask2former 到 segmentation/configs/mask2former/
mkdir -p segmentation/local_configs/mask2former
mkdir -p segmentation/pretrained
# Swin-T COCO 预训练（示例 URL 以 mmdet 文档为准）
# wget -O segmentation/pretrained/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic.pth ...
```

**验收**：`python -c "import mmdet"` 无报错；config 目录存在。

**注意**：

- 若 mmcv 与现有 mmseg 0.30 **版本冲突**，建独立 env `segman_mmdet` 或在 Docker 层隔离。
- **禁止** 覆盖 SegMAN v2@6k 训练环境；F1 推理阶段再与 E2E 合并。

---

##### F1-2 — 编写训练 config（0.5 天，需开发）

**目的**：将 Trans10K 伪实例 COCO 接入 Mask2Former 训练管线。


| 项               | 内容                                                                        |
| --------------- | ------------------------------------------------------------------------- |
| **新建**          | `segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py` |
| **数据集**         | `CocoDataset`，ann=`coco_instances/train.json`，img=`img_dir/train`         |
| **类别**          | 11 前景类（category_id 1～11，与 `CLASSES` 一致）                                   |
| **num_queries** | **100**（首轮）；二轮可调 **150**                                                  |
| **input**       | 短边 512～640，保持 Trans10K 宽高比                                                |


**config 关键字段（草稿）**：

```python
# 数据
data_root = 'data/trans10k'
train_ann = 'coco_instances/train.json'
val_ann = 'coco_instances/val.json'
metainfo = dict(classes=(
    'box', 'bottle', 'window', 'eyeglass', 'freezer',
    'jar_kettle', 'door', 'cup', 'wall', 'bowl', 'shelf'))

# 模型
model = dict(
    type='Mask2Former',
    num_queries=100,
    panoptic_head=dict(num_things_classes=11, num_stuff_classes=0),
    ...
)

# 训练
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=5)
# 或 IterBased: max_iters=40000
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05))
```

**验收**：`python tools/train.py ... --cfg-options train_cfg.max_iters=10` smoke 10 iter 无报错。

**config 模板来源**（任选）：

```bash
# 从 mmdet model zoo 复制并改 dataset
# 参考：mmdetection/configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic.py
# 改 num_things_classes=11, num_stuff_classes=0, data_root, ann_file
```

**与 E4 MaskRCNN MVP 的差异**（单变量原则）：


| 项        | E4 MVP                 | F1                    |
| -------- | ---------------------- | --------------------- |
| 框架       | torchvision Mask R-CNN | **mmdet Mask2Former** |
| 训练数据     | 1500 图 × 5 epoch       | **5000 图 × 40k iter** |
| 实例头      | FPN + RoI              | **query-based mask**  |
| 实测 match | 29.28%                 | 目标 **≥65%**           |


---

##### F1-3 — 全量训练（1～2 周）

**目的**：COCO 预训练初始化，全量 5000 train 伪实例，**40k iter**。

```bash
cd /workspace/segman/segmentation
source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman

# 单卡（RTX 4060 建议 batch=2）
python tools/train.py \
  local_configs/mask2former/m2f_trans10k_pseudo_instances.py \
  --work-dir outputs/m2f_trans10k_pseudo \
  --cfg-options train_dataloader.batch_size=2

# 多卡
# bash tools/dist_train.sh \
#   local_configs/mask2former/m2f_trans10k_pseudo_instances.py 2 \
#   --work-dir outputs/m2f_trans10k_pseudo
```

**训练监控**：


| 项    | 建议                                  |
| ---- | ----------------------------------- |
| loss | 总 loss 持续下降；过拟合看 val AP             |
| ckpt | 每 5k iter 存 `iter_5000.pth` …       |
| 早停   | **每 5k** 触发 F1-4 分割 match（见下）       |
| 日志   | `outputs/m2f_trans10k_pseudo/*.log` |


**验收（训练中期）**：

- iter **5000**：loss 明显低于初始（参考 <2.0）
- iter **20000**：F1-4 实例 match **≥55%**，否则 F1-SEG-FAIL 停训调参

---

##### F1-4 — val 推理 + 分割级 instance match（0.5 天）

**目的**：在 **不接分类** 的情况下，仅用 mask 质量评估 checkpoint，决定早停与最优 ckpt。

**Step A：批量推理 val**

```bash
cd /workspace/segman/segmentation

python tools/test.py \
  local_configs/mask2former/m2f_trans10k_pseudo_instances.py \
  outputs/m2f_trans10k_pseudo/iter_20000.pth \
  --work-dir outputs/m2f_trans10k_pseudo/infer_iter20000

# 导出 COCO pred JSON（需在 test.py 或自定义脚本中加 --out pred.json）
python tools/infer_m2f_export_coco.py \
  --config local_configs/mask2former/m2f_trans10k_pseudo_instances.py \
  --checkpoint outputs/m2f_trans10k_pseudo/iter_20000.pth \
  --ann segmentation/data/trans10k/coco_instances/val.json \
  --out-dir outputs/m2f_trans10k_pseudo/infer_iter20000
```

**Step B：分割级 match（待建 `eval_instance_match.py`）**

```bash
python segmentation/tools/eval_instance_match.py \
  --pred-coco outputs/m2f_trans10k_pseudo/infer_iter20000/pred_instances.json \
  --gt-coco segmentation/data/trans10k/coco_instances/val.json \
  --iou-match 0.25 \
  --out outputs/e2e_improve/f1_seg_match_iter20000.json
```

**验收**：


| iter       | 实例 match 目标               |
| ---------- | ------------------------- |
| 5k         | ≥50%                      |
| 20k        | **≥58%**                  |
| 40k / best | **≥62%**（才进入 F1-5 全量 E2E） |


**待建工具规格**：


| 工具                         | 路径                    | 功能                                                                                                                        |
| -------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `infer_m2f_export_coco.py` | `segmentation/tools/` | 加载 mmdet config + ckpt，对 val 图批量推理，输出 COCO `pred_instances.json`（含 `segmentation` RLE、`category_id`、`score`）              |
| `eval_instance_match.py`   | `segmentation/tools/` | 读 pred/GT COCO JSON，按 **IoU≥0.25 greedy** 匹配（与 E2E `--iou-match 0.25` 一致），输出 `match_rate`、`pred_gt_ratio`、per-class match |


`**eval_instance_match.py` 输出示例**：

```json
{
  "match_rate": 0.58,
  "pred_gt_ratio": 1.02,
  "num_gt": 3105,
  "num_pred": 3167,
  "per_class": {"wall": {"match_rate": 0.48, "miss": 420}, "door": {"match_rate": 0.47}}
}
```

`**run_f1_gate_check.py` 规格**（对齐 `run_e_gate_check.py`）：


| 项       | 内容                                         |
| ------- | ------------------------------------------ |
| 对照 runs | `f1_b1_ref`（semantic）、`f1_m2f_e2e`（m2f）    |
| 闸门      | F1-PASS-1～5（§4.1 验收表）                      |
| F1-FAIL | match **<62%**                             |
| 输出      | `outputs/e2e_improve/f1_plan_summary.json` |


---

##### F1-5 — E2E 接入（1～2 天，需开发）

**目的**：用 Mask2Former 输出替换语义 CC，**分类栈 + B1 后处理不变**。


| 项       | 内容                                                                                                   |
| ------- | ---------------------------------------------------------------------------------------------------- |
| **新建**  | `transgrasp/pipelines/m2f_predictor.py` 或在 `instance_predictor.py` 增加 `Mask2FormerInstancePredictor` |
| **修改**  | `segment_and_classify.py`：`--instance-source m2f`，`--m2f-config`，`--m2f-checkpoint`                  |
| **接口**  | `predict_instances(rgb) -> list[InstanceROI]`，与 maskrcnn 后端一致                                        |
| **后处理** | 仍走 `postprocess_instances`（B1 NMS / min_area）                                                        |


**推理逻辑**：

```text
rgb → Mask2Former forward → N×(class, mask, score)
  → score ≥ 0.3 过滤
  → 转 InstanceROI（mask/bbox/class_id）
  → B1 postprocess（min_area128, nms_iou0.5）
  → classify_instances（OpenCLIP + reject）
```

**Smoke test（50 张）**：

```bash
python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images 50 \
  --instance-source m2f \
  --m2f-config segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py \
  --m2f-checkpoint segmentation/outputs/m2f_trans10k_pseudo/best_bbox_mAP.pth \
  --out-dir outputs/e2e_improve/f1_m2f_smoke50 \
  --min-area 128 --nms-iou 0.5 --iou-match 0.25 --min-area-shelf 32
```

**验收**：smoke 50 张无 crash；pred 实例数 **0.8～1.2 × GT**；match **高于 B1**。

---

##### F1-6 — 全量 E2E 验收 + E2-0 审计（0.5～1 天）

**目的**：最优 ckpt 跑全量 val，产出 `f1_plan_summary.json` 与 F1-PASS 判定。

```bash
# 全量 E2E
python transgrasp/pipelines/segment_and_classify.py \
  --eval-split val --max-images -1 \
  --instance-source m2f \
  --m2f-config segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py \
  --m2f-checkpoint segmentation/outputs/m2f_trans10k_pseudo/best_bbox_mAP.pth \
  --out-dir outputs/e2e_improve/f1_m2f_e2e \
  --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10 \
  --iou-match 0.25 --min-area-shelf 32 --match-algorithm greedy

python transgrasp/pipelines/summarize_e2e_eval.py \
  --eval-dir outputs/e2e_improve/f1_m2f_e2e

# E2-0 根因审计
python transgrasp/pipelines/export_unmatched_instances.py \
  --eval-dir outputs/e2e_improve/f1_m2f_e2e \
  --out-dir outputs/e2e_improve/e2_audit_f1_best \
  --sample-wall 100 --sample-door 50

# 闸门判定
python transgrasp/pipelines/run_f1_gate_check.py
```

**一键脚本（待建 `scripts/run_plan_f1.sh`）**：

```bash
bash scripts/run_plan_f1.sh 2>&1 | tee outputs/e2e_improve/f1_plan_run.log
```

**脚本骨架**（对齐 `scripts/run_plan_e.sh` 结构）：

```bash
#!/usr/bin/env bash
# Scheme F1: Mask2Former full train + seg match + E2E eval
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate segman 2>/dev/null || true

IMPROVE=outputs/e2e_improve
M2F_CFG=segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py
M2F_WD=segmentation/outputs/m2f_trans10k_pseudo
M2F_CKPT="${M2F_CKPT:-${M2F_WD}/best_bbox_mAP.pth}"

B1_ARGS=(
  --eval-split val --max-images -1
  --min-area 128 --nms-iou 0.5 --max-aspect-ratio 10
  --iou-match 0.25 --min-area-shelf 32 --match-algorithm greedy
)

run_eval() {
  local name="$1"; shift
  echo "========== ${name} =========="
  python transgrasp/pipelines/segment_and_classify.py \
    --out-dir "${IMPROVE}/${name}" "${B1_ARGS[@]}" "$@"
  python transgrasp/pipelines/summarize_e2e_eval.py --eval-dir "${IMPROVE}/${name}"
}

# F1-0 基线 + COCO
python segmentation/tools/export_trans10k_coco_instances.py \
  --data-root segmentation/data/trans10k --splits train,val \
  --min-area 64 --out-dir segmentation/data/trans10k/coco_instances

run_eval f1_b1_ref --instance-source semantic \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth

# F1-3 训练（SKIP_TRAIN=1 可跳过）
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  python segmentation/tools/train.py "${M2F_CFG}" \
    --work-dir "${M2F_WD}" \
    --cfg-options train_dataloader.batch_size="${BATCH_SIZE:-2}"
fi

# F1-4 分割级 match（每 5k ckpt 或 best）
for IT in 5000 10000 15000 20000 25000 30000 35000 40000; do
  CK="${M2F_WD}/iter_${IT}.pth"
  [[ -f "${CK}" ]] || continue
  python segmentation/tools/infer_m2f_export_coco.py \
    --config "${M2F_CFG}" --checkpoint "${CK}" \
    --ann segmentation/data/trans10k/coco_instances/val.json \
    --out-dir "${M2F_WD}/infer_iter${IT}"
  python segmentation/tools/eval_instance_match.py \
    --pred-coco "${M2F_WD}/infer_iter${IT}/pred_instances.json" \
    --gt-coco segmentation/data/trans10k/coco_instances/val.json \
    --iou-match 0.25 \
    --out "${IMPROVE}/f1_seg_match_iter${IT}.json"
done

# F1-6 E2E（需 M2F_CKPT 存在）
if [[ -f "${M2F_CKPT}" ]]; then
  run_eval f1_m2f_e2e \
    --instance-source m2f \
    --m2f-config "${M2F_CFG}" \
    --m2f-checkpoint "${M2F_CKPT}"
fi

python transgrasp/pipelines/run_f1_gate_check.py

BEST=$(python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/e2e_improve/f1_plan_summary.json')
if p.is_file():
    b = json.loads(p.read_text()).get('best_run') or {}
    print(b.get('name') or 'f1_b1_ref')
else:
    print('f1_b1_ref')
PY
)

python transgrasp/pipelines/export_unmatched_instances.py \
  --eval-dir "${IMPROVE}/${BEST}" \
  --out-dir "${IMPROVE}/e2_audit_f1_best" \
  --sample-wall 100 --sample-door 50 2>/dev/null || true

echo "Scheme F1 done. See ${IMPROVE}/f1_plan_summary.json (best=${BEST})"
```

**环境变量**：


| 变量           | 默认                  | 说明                              |
| ------------ | ------------------- | ------------------------------- |
| `SKIP_TRAIN` | `0`                 | `1` 时跳过 F1-3，仅跑已有 ckpt 的 F1-4/6 |
| `BATCH_SIZE` | `2`                 | RTX 4060 单卡建议 2                 |
| `M2F_CKPT`   | `best_bbox_mAP.pth` | 指定 E2E 用 checkpoint             |


---

**台账字段**（`outputs/e2e_improve/f1_plan_summary.json`）：

```json
{
  "baseline_match": 0.5916,
  "oracle_match": 0.9298,
  "runs": [
    {"name": "f1_b1_ref", "match_rate": 0.5916, "instance_source": "semantic"},
    {"name": "f1_m2f_e2e", "match_rate": null, "checkpoint": "best_bbox_mAP.pth"}
  ],
  "best_run": null,
  "F1_PASS": false,
  "deploy_instance_source": null
}
```

---

##### F1-7 — 未 PASS 调参二轮（3～5 天）

**目的**：F1-FAIL 时单变量 grid，避免重复整 pipeline 失败。

**推荐 grid（每次只改一项）**：


| 编号    | 变量                    | 候选值                    | 针对               |
| ----- | --------------------- | ---------------------- | ---------------- |
| F1-7a | `num_queries`         | 100 / **150** / 200    | 欠预测 / miss       |
| F1-7b | score_thresh          | 0.25 / **0.30** / 0.35 | pred/GT 平衡       |
| F1-7c | wall/door loss weight | 1.0 / **1.5** / 2.0    | 结构类 recall       |
| F1-7d | max_iters             | 40k / **60k**          | 欠拟合              |
| F1-7e | input scale           | 512 / **640**          | 小目标 shelf/window |


**命令模板**：

```bash
for NQ in 100 150; do
  python tools/train.py \
    local_configs/mask2former/m2f_trans10k_pseudo_instances.py \
    --work-dir "outputs/m2f_trans10k_q${NQ}" \
    --cfg-options model.num_queries=${NQ}
done
```

**终止条件**：

- 任一 run **F1-PASS** → 停止 grid，写 deploy
- 全部 **match <62%** → **进入 F2**（§5），不再加第三轮 grid

---

##### F1 代码改动清单


| 文件                                                                        | 状态   | 改动                                |
| ------------------------------------------------------------------------- | ---- | --------------------------------- |
| `segmentation/tools/export_trans10k_coco_instances.py`                    | ✅ 已有 | E0 伪实例                            |
| `segmentation/tools/browse_coco_instances.py`                             | ✅ 已有 | 目视 QA                             |
| `segmentation/tools/eval_instance_match.py`                               | ❌ 待建 | F1-4 分割 match                     |
| `segmentation/tools/infer_m2f_export_coco.py`                             | ❌ 待建 | val 批量导出 pred                     |
| `segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py` | ❌ 待建 | F1-2 训练 config                    |
| `transgrasp/pipelines/instance_predictor.py`                              | ⚠ 部分 | 增加 `Mask2FormerInstancePredictor` |
| `transgrasp/pipelines/segment_and_classify.py`                            | ⚠ 部分 | `--instance-source m2f` CLI       |
| `transgrasp/pipelines/run_f1_gate_check.py`                               | ❌ 待建 | F1-PASS 判定                        |
| `scripts/run_plan_f1.sh`                                                  | ❌ 待建 | F1-0～F1-6 一键                      |


---

##### F1 实现状态一览


| 编号   | 步骤         | 代码       | 执行                 |
| ---- | ---------- | -------- | ------------------ |
| F1-0 | 基线 + COCO  | ✅ export | ✅ val=3105         |
| F1-1 | mmdet 环境   | ❌        | 未装 mmdet           |
| F1-2 | m2f config | ❌        | —                  |
| F1-3 | 40k 训练     | ❌        | —                  |
| F1-4 | 分割 match   | ❌        | —                  |
| F1-5 | E2E 接入 m2f | ❌        | maskrcnn 已有，m2f 待接 |
| F1-6 | 全量验收       | ❌        | —                  |
| F1-7 | 调参 grid    | ❌        | —                  |


**建议开发顺序（1 周 MVP 接入 + 2 周训练）**：

```text
Day 1   F1-1 环境 + F1-2 config smoke
Day 2   F1-4 eval_instance_match + infer export 脚本
Day 3   F1-5 m2f 后端 + CLI smoke
Day 4～18  F1-3 训练 40k（并行每 5k 跑 F1-4）
Day 19  F1-6 全量 E2E + 审计
Day 20  未 PASS → F1-7 grid 或 F2
```

---

##### F1 风险与对策


| 风险                 | 对策                           |
| ------------------ | ---------------------------- |
| mmdet 与 mmseg 版本冲突 | 独立 conda env `segman_mmdet`  |
| 伪实例 GT 噪声          | F1-4 对比 oracle；F2-A 真标注校准    |
| wall 仍 miss        | F1-7c class weight；F2-B 专项   |
| pred/GT <0.95 欠预测  | ↑ num_queries；↓ score_thresh |
| pred/GT >1.08 过分割  | ↑ score_thresh；B1 NMS 保持     |
| 训练 2 周仍 <62%       | 不拖延，直接 F2-D 混合架构             |


---

##### F1 与 B / E 的关系


| 组件       | B1 语义  | E4 MaskRCNN MVP | **F1 Mask2Former** |
| -------- | ------ | --------------- | ------------------ |
| 实例来源     | 语义 CC  | 检测头             | **query mask**     |
| 训练量      | v2@6k  | 1500×5ep        | **5000×40k iter**  |
| 实测 match | 59.16% | 29.28% ❌        | 目标 **≥65%**        |
| 针对 miss  | 否      | intended 是      | **是（主目标）**         |


**单变量原则**：F1 相对 E4 的区别是 **模型架构（Mask2Former）+ 训练规模（全量 40k）**；不要同时改 E2E 后处理口径。

---

### 4.2 依据（摘要）

- GT oracle **92.98%**：实例级 mask 质量是杠杆支点。
- 语义 CC 派生实例有 **结构性欠分割**（大 wall CC 吃匹配）。
- Mask2Former 直接输出 query-based 实例 mask，针对 **miss + 欠分割**。

### 4.3 若 F1 仍不足 → 进入 F2

- 提高 `num_queries`（100 → 150）— 见 F1-7
- wall/door **class-weighted loss**
- 启动 **F2-B** wall/door 专项或 **F2-D** 混合架构（§5）

---

## 5. F2 — 结构性突破（1～2 月）

**定位**：F1 未达 65% 或需冲击 **75%+** 时启动。

### 5.1 F2-A 标注升级


| 项   | 内容                                    |
| --- | ------------------------------------- |
| 问题  | 当前 GT 实例 = 语义图 CC 派生，与真实物体实例不完全一致     |
| 做法  | 对 val 子集（200 张）做 **真 instance id** 标注 |
| 收益  | 可训练/评测 **真实实例分割**；消除伪 GT 噪声           |
| 工期  | 2～4 周（标注）+ 1 周（训练）                    |


### 5.2 F2-B wall/door 专项


| 项   | 内容                                                                                 |
| --- | ---------------------------------------------------------------------------------- |
| 问题  | wall+door 占未匹配 **81%**，miss 主因                                                     |
| 做法  | ① 单独 copy-paste door 到 wall 场景；② boundary loss 权重↑；③ hard example mining（未匹配样本重采样） |
| 注意  | 方案 C 短训 FAIL；需 **更长 schedule + 实例 match 早停**，非 mIoU alone                          |
| 目标  | wall match **55% → 65%**                                                           |


### 5.3 F2-C SAM 弱 prompt 二阶段

```text
SegMAN 语义 → CC bbox → SAM2 点/bbox prompt → refined mask → E2E
```


| 阶段          | 命令/目的                           |
| ----------- | ------------------------------- |
| E2-0 oracle | GT 中心点 prompt，测上界（目标 75%+）      |
| E2-1 weak   | 语义 CC bbox prompt，测全自动（目标 65%+） |


**代价**：推理 latency 高；可作为 F1 补充实验，验证 mask refine 上限。

### 5.4 F2-D 检测 + 实例分割混合


| 组件   | 类                           | 模型                   |
| ---- | --------------------------- | -------------------- |
| 物体分支 | bottle, cup, box, …         | YOLOv8-seg / RT-DETR |
| 结构分支 | wall, door, window          | Mask2Former          |
| 融合   | `hybrid_instance_fusion.py` | NMS + 类优先级           |


**预期**：物体类 match **85%+**；全类 **≥F1 alone**。

---

## 6. 路线选型决策树

```text
有 2～4 周 GPU？
├─ 是 → F1（Mask2Former 全量长训）  ← 默认路径
│       ├─ F1 PASS（match≥65%）→ 更新实例源，继续调参冲 70%
│       └─ F1 FAIL（match<62%）→ F2-B wall/door 专项 或 F2-D 混合
└─ 否 → 缩小 F1（减 iter / 子集验证）或并行推进 F2-A 标注立项

F1 + F2-B 仍 <65%？
└─ 是 → F2-A 真 instance 标注 + 重训
```

---

## 7. 阶段目标


| 阶段           | match_rate | strict E2E | 说明          |
| ------------ | ---------- | ---------- | ----------- |
| **现状 B1**    | 59.16%     | 50.05%     | 基线          |
| D4 TTA（参考）   | 60.48%     | 51.37%     | 语义路线小幅增量    |
| **F1 目标**    | **≥65%**   | **≥55%**   | Mask2Former |
| F1 stretch   | 70%        | 58%        | 长训 + 调参     |
| F2 stretch   | 75%+       | 62%+       | 标注 + 混合     |
| GT oracle 上界 | 92.98%     | 73.37%     | 理论上限参考      |


---

## 8. 资源与风险

### 8.1 算力估算


| 任务             | GPU·时间                    | 备注       |
| -------------- | ------------------------- | -------- |
| F1 Mask2Former | **1.5～2 周** × 1×RTX 4060+ | 40k iter |
| F2 标注          | 人力 2～4 周                  | 与 GPU 并行 |
| F2-C SAM       | +50～200ms/图               | 实验用      |


### 8.2 风险


| 风险            | 对策                       |
| ------------- | ------------------------ |
| F1 仍 <62%     | 进入 F2-B / F2-D           |
| 伪实例 GT 噪声     | F2-A 真标注子集校准             |
| wall/door 本质难 | F2-B 专项 + 提高 num_queries |
| mmdet 环境冲突    | 独立 conda env 或 Docker 层  |


---

## 9. 复现与产物索引


| 类型               | 路径                                                                        |
| ---------------- | ------------------------------------------------------------------------- |
| 根因与 B～E 手册       | `docs/E2E_实例匹配偏低根因与改进方案.md`                                               |
| B 台账             | `outputs/e2e_improve/b_plan_summary.json`                                 |
| C 台账             | `outputs/e2e_improve/c_plan_summary.json`                                 |
| D 台账             | `outputs/e2e_improve/d_plan_summary.json`                                 |
| E 台账             | `outputs/e2e_improve/e_plan_summary.json`                                 |
| COCO 伪实例         | `segmentation/data/trans10k/coco_instances/`                              |
| F1 训练 config（待建） | `segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py` |
| F1 分割 match（待建）  | `segmentation/tools/eval_instance_match.py`                               |
| F1 计划脚本（待建）      | `scripts/run_plan_f1.sh`                                                  |
| F1 台账（待填）        | `outputs/e2e_improve/f1_plan_summary.json`                                |
| 实例预测器            | `transgrasp/pipelines/instance_predictor.py`                              |
| E2E 回归           | `scripts/run_e2e_regression.sh`                                           |


---

## 10. 推荐行动（按优先级）


| 优先级    | 行动                                    | 工期    |
| ------ | ------------------------------------- | ----- |
| **P0** | F1-1～F1-2：mmdet 环境 + m2f config smoke | 1～2 天 |
| **P1** | F1-3：40k iter 全量训练 + 每 5k F1-4 早停     | 1～2 周 |
| **P2** | F1-5～F1-6：m2f E2E 接入 + 全量验收           | 2～3 天 |
| P3     | F1-7 调参 grid 或 F2-B / F2-D            | 1～2 周 |
| P4     | F2-A 真 instance 标注                    | 1～2 月 |


---

## 11. 修订记录


| 版本   | 日期         | 说明                                     |
| ---- | ---------- | -------------------------------------- |
| v1.0 | 2026-05-27 | 首版：整合 B～E 实验结论，给出 F0/F1/F2 后续路线        |
| v1.1 | 2026-05-27 | 移除 F0 结题/交付路线；F1 为当前主攻                 |
| v1.2 | 2026-05-27 | 新增 §4.1 F1 完整执行手册（F1-0～F1-7、脚本骨架、工具规格） |


