# 路线 B 正式交付 — balanced-v2 @ iter_6000

| 项目 | 内容 |
|------|------|
| 文档版本 | v2.0（交付权重自 fix5k 升级为 v2@6k） |
| 日期 | 2026-05-19 |
| **正式分割权重** | `outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth` |
| **正式配置** | `local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py` |
| **test 指标（val 1000）** | mIoU **81.80%**（+0.96 vs fix5k）；bowl **80.70%**；window **77.16%**；shelf **67.73%**（≈基线） |
| **历史对照** | fix5k `iter_5000.pth`（80.84% mIoU）— 软著/专利可作从属实施例 |
| **实验归档（不部署）** | `balanced10k/*`、`bowl5k/*`、`v2/iter_8000` 等 — 见对比分析文档 |

**工作目录约定**：Docker 内 `cd /workspace/segman/segmentation`；Windows 本地将路径改为 `segmentation\` 下对应相对路径。

---

## 总览：还剩什么

```text
[已完成] 路线 A 环境 + 基线 80k
[已完成] 路线 B 阶段 1～3 + fix5k + balanced-v2（8k 扫完）
[已完成] v2 终选：**iter_6000** 作为正式交付（mIoU/bowl/shelf 均衡）
────────────────────────────────────────────
[进行中] ① **v2@6k** 交付固化（评测 json ✓、可视化/冻结清单待做）
[待做] ② B8 分割推理 API（transgrasp，加载 v2@6k）
[待做] ③ 路线 C（见《路线C_细分类与抓取实施步骤.md》，分割侧用 v2@6k）
[待做] ④ 文档与申报收尾（主表报 v2@6k vs 基线；fix5k 作对照）
```

---

## 阶段 ① v2@6k 交付固化（约 0.5～1 天）

### 步骤 ①-1：复现正式 test 数字

**目的**：确认权重与配置可一键复现交付指标，作为验收与申报材料依据。

```bash
cd /workspace/segman/segmentation

python tools/test.py \
  local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py \
  --checkpoint outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --eval mIoU
```

**验收**：Summary 中 mIoU ≈ **81.80**；bowl ≈ **80.70**；window ≈ **77.16**；shelf ≈ **67.73**。

**已归档（2026-05-24）**：`outputs/trans10k_lass_mmscope_balanced_v2/eval_deliver_6k/eval_single_scale_20260524_133106.json` — mIoU **0.8180**，与 §12.6 一致。

---

### 步骤 ①-2：导出评测 json + 与基线对比表

**目的**：生成可归档的数值文件；论文/答辩直接引用 Δ 表。

```bash
python tools/test.py \
  local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py \
  --checkpoint outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --eval mIoU \
  --work-dir outputs/trans10k_lass_mmscope_balanced_v2/eval_deliver_6k

python scripts/compare_miou_vs_baseline.py \
  "outputs/trans10k_lass_mmscope_balanced_v2/eval_deliver_6k/eval_single_scale_*.json"
```

**验收**：终端打印 12 类 ↑/↓/≈；json 保存在 `eval_deliver_6k/`。

**已完成**：`eval_deliver_6k/eval_single_scale_20260524_133106.json`（config = `segman_b_trans10k_lass_balanced_v2.py`）。

---

### 步骤 ①-3：可视化（mask / 边界）

**目的**：软著、专利附图、报告「window / bowl 改善」目视证据。

```bash
python tools/test.py \
  local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py \
  --checkpoint outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --eval mIoU \
  --show-dir outputs/trans10k_lass_mmscope_balanced_v2/vis_deliver_6k
```

**验收**：`vis_deliver/` 下有 val 叠加图；另挑含 **window / bowl** 的图各 3～5 张作报告插图。

---

### 步骤 ①-4：自检 ignore_index 与模块（可选）

**目的**：防止后续改代码后 pad 标签再次污染训练/评测。

```bash
python scripts/verify_ignore_index_fix.py
python scripts/verify_mmscope_phase2.py
```

**验收**：脚本退出码 0，无报错。

---

### 步骤 ①-5：冻结「交付清单」文件

**目的**：路线 C、协作、答辩统一只引用下列路径，避免误用 balanced10k / 80k 失败权重。

| 类型 | 路径 |
|------|------|
| **分割权重（正式）** | `outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth` |
| **训练/推理配置（正式）** | `local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py` |
| 架构基类（勿为 v2 乱改） | `segman_b_trans10k_lass.py`（LASS+MMSCopE 结构） |
| fix5k 历史备份 | `recipes/segman_b_trans10k_lass_fix5k.py` + `fix5k/iter_5000.pth` |
| v2 其他 ckpt | `iter_4000`（window 最高）、`iter_8000`（**勿部署**） |
| 对比文档 | `路线B_基线_fix5k_balanced10k_对比分析.md` §5.8、§5.11 |
| 数值来源 | 《路线B_平衡微调方案.md》§12.6 |

**验收**：在 `transgrasp/configs/`（阶段 ② 创建）中写入相同 checkpoint 路径。

---

## 阶段 ② B8：分割推理 API（约 2～3 天）

### 步骤 ②-1：创建 transgrasp 目录

**目的**：与《项目实施步骤指南.md》路线 C 结构对齐；分割与抓取解耦。

```bash
cd /workspace/segman
mkdir -p transgrasp/segmentation transgrasp/grasping transgrasp/ui transgrasp/configs transgrasp/pipelines
```

---

### 步骤 ②-2：实现 `infer_segman.py`

**目的**：路线 C 只依赖该接口；内部固定加载 **v2@6k** + `segman_b_trans10k_lass_balanced_v2.py`。

**新建**：`transgrasp/segmentation/infer_segman.py`（接口见《项目实施步骤指南.md》B8）

**建议配置** `transgrasp/configs/segman_route_b.yaml`：

```yaml
config: local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py
checkpoint: outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth
device: cuda:0
```

**验收脚本示例**：

```bash
cd /workspace/segman/segmentation
python -c "
import sys; sys.path.insert(0, '..')
from transgrasp.segmentation.infer_segman import SegmanPredictor
import cv2
p = SegmanPredictor(
    config='local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py',
    checkpoint='outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth',
)
img = cv2.imread('data/trans10k/images/val/<某张>.jpg')
out = p.predict(img)
assert out['sem_seg'].shape[:2] == img.shape[:2]
print('classes', out['sem_seg'].min(), out['sem_seg'].max())
"
```

**验收**：输出 `(H,W)` 语义图，类别 0～11；可选 `mask_union`（非背景并集）供抓取用。

---

### 步骤 ②-3：单张推理与基线/v2@6k 目视对比（可选）

**目的**：抓取前确认 mask 边界可用（尤其 **bowl / shelf**）。

```bash
# 使用 ②-2 API，对同一张 val 图输出 sem_seg 上色图
# 与 outputs/trans10k_lass_mmscope_balanced_v2/vis_deliver_6k 一致即可
```

---

## 阶段 ③ 路线 C：抓取仿真 + UI（约 2～4 周）

> **前提**：阶段 ①、② 完成；**全程使用 v2@6k 权重**，勿误用 fix5k / iter_8000 / balanced10k。

### 步骤 ③-1：PyBullet 环境

**目的**：空载加载机械臂/桌面 URDF，为仿真打底。

```bash
pip install pybullet
# 新建 transgrasp/grasping/pybullet_env.py
python transgrasp/grasping/pybullet_env.py   # 或冒烟脚本
```

**验收**：GUI/无头模式能步进关节，无 URDF 路径错误。

---

### 步骤 ③-2：ASGrasp 或启发式抓取封装

**目的**：输入 **mask + class_id** → 输出 6D 抓取位姿。

**新建**：

- `transgrasp/grasping/asgrasp_wrapper.py`
- `transgrasp/configs/grasp_class_prior.yaml`（每类 approach 距离、夹爪宽等）

**验收**：给定 v2@6k 导出的二值 mask，能返回合法 `4x4` 或 `pos+quat`。

---

### 步骤 ③-3：单次抓取闭环

**目的**：验证「分割 → 选实例 → 规划 → PyBullet 执行」端到端。

**新建**：`transgrasp/grasping/run_grasp_sim_once.py`

```bash
cd /workspace/segman
python transgrasp/grasping/run_grasp_sim_once.py \
  --image data/trans10k/images/val/<sample>.jpg \
  --checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth
```

**验收**：至少 1 个场景完成 approach → close → lift；失败有日志。

---

### 步骤 ③-4：抓取参数搜索（可选）

**目的**：提高仿真成功率；对应设计书 G0/G1。

**新建**：`transgrasp/grasping/optimize_grasp.py`

**验收**：同一物体上，优化后成功率 ≥ 默认参数。

---

### 步骤 ③-5：Gradio UI

**目的**：演示「上传图 → v2@6k 分割 → 点选 → 仿真抓取」。

```bash
pip install gradio
python transgrasp/ui/app.py
```

**验收**：非开发者可完成全流程；UI 读 `segman_route_b.yaml`（指向 v2@6k）。

---

### 步骤 ③-6：端到端批量表（10～20 张）

**目的**：结题/论文「分割 + 抓取」联合指标。

| 列 | 内容 |
|----|------|
| 图像 id | val 文件名 |
| mIoU / 类 IoU | 与 GT 对比（可选） |
| mask 质量 | 目视 0/1 |
| 仿真成功 | 0/1 |

**验收**：表格可复现；README 记录命令与 v2@6k 路径。

---

## 阶段 ④ 文档与申报收尾（与阶段 ③ 并行）

### 步骤 ④-1：更新项目主文档中的「正式权重」

**目的**：全文一致写 **v2@6k**，避免误用 fix5k / iter_8000。

| 文件 | 动作 |
|------|------|
| `路线C_细分类与抓取实施步骤.md` | 分割 checkpoint 改为 v2@6k |
| `路线B_LASS_MMSCopE_实施清单.md` | 正式权重改为 v2@6k |
| `Trans10K_SegMAN_B_训练与评测结果.md` | 增「路线 B 正式交付 v2@6k」一节 |

---

### 步骤 ④-2：软著 / 专利材料

**目的**：申报材料与运行版本一致。

| 材料 | 建议内容 |
|------|----------|
| **软著** | 源码含 LASS/MMSCopE + BowlAntiCupLoss；说明书用 **balanced-v2 @ 6k** 训练/推理流程 |
| **专利** | **主实施例**：基线 vs **v2@6k**（mIoU 81.80，bowl/shelf 均衡）；fix5k、v2@4k 可作从属/对照 |

---

### 步骤 ④-3：论文/答辩表格（固定一张主表）

**目的**：主文报 **v2@6k vs 基线**；fix5k、v2@4k、balanced10k 放消融。

**主表数据**：对比分析 §5.8（6k 行）+ §2 基线列。

**一句话贡献**：LASS+MMSCopE + balanced-v2 微调，mIoU **81.80%**（+1.09 vs 基线），bowl **80.70%**，shelf **≈基线**，相对 fix5k mIoU **+0.96%**。

---

## 不再执行的项（已决策冻结）

| 项 | 原因 |
|----|------|
| balanced10k / fix5k 作默认部署 | 已由 **v2@6k** 替代；fix5k 仅历史对照 |
| v2 `iter_4000` / `iter_8000` 默认部署 | 4k shelf 低；8k 长训退化 |
| bowl5k、bowl3k_from10k 再训 | 未达 mIoU+bowl 双达标或已实验失败 |
| fix80k / 80k 无修复权重 | mIoU 塌缩，见实施清单 §0.2 |
| 从 iter_10000 再微调 bowl | 1b 三次 test 未达标，见平衡微调 §11 |

消融实验**保留在仓库与对比文档中**，写论文「补充实验」即可，**不进入 transgrasp 默认配置**。

---

## 命令速查（正式：v2@6k）

```bash
# balanced-v2 训练（已完成；复现见 scripts/train_route_b_balanced_v2.sh）
# 正式评测
python tools/test.py local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py \
  --checkpoint outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --eval mIoU

# 路线 C / transgrasp：config + checkpoint 与上一致
```

**fix5k 对照（非交付）**：

```bash
python tools/test.py local_configs/segman_trans/segman_b_trans10k_lass.py \
  --checkpoint outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth --eval mIoU
```

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-23 | v1.0：以 fix5k 为正式权重 |
| 2026-05-19 | **v2.0：正式交付升级为 balanced-v2 `iter_6000.pth`** |
