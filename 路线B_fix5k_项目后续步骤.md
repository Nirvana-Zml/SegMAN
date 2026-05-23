# 路线 B 以 fix5k 为准 — 项目后续完成步骤

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.0 |
| 日期 | 2026-05-23 |
| **正式分割权重** | `outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth` |
| **正式配置** | `local_configs/segman_trans/segman_b_trans10k_lass.py` |
| **test 指标（val 1000）** | mIoU **80.84%**（+0.13 vs 基线）；window **76.27%**（+9.65）；bowl **80.07%**（+1.16） |
| **实验归档（不再作主推）** | `balanced10k/*`、`bowl5k/*`、`bowl3k_from10k/*` — 见《路线B_基线_fix5k_balanced10k_对比分析.md》 |

**工作目录约定**：Docker 内 `cd /workspace/segman/segmentation`；Windows 本地将路径改为 `segmentation\` 下对应相对路径。

---

## 总览：还剩什么

```text
[已完成] 路线 A 环境 + 基线 80k
[已完成] 路线 B 阶段 1～3 + fix5k 训练与 test
[已完成] balanced / bowl 消融（归档，不部署）
────────────────────────────────────────────
[待做] ① fix5k 交付固化（评测、可视化、权重清单）
[待做] ② B8 分割推理 API（transgrasp 用）
[待做] ③ 路线 C（细分类 + ASGrasp + PyBullet，见《路线C_细分类与抓取实施步骤.md》）
[待做] ④ 文档与申报收尾（软著/专利/论文表）
```

---

## 阶段 ① fix5k 交付固化（约 0.5～1 天）

### 步骤 ①-1：复现官方 test 数字

**目的**：确认权重与配置可一键复现 mIoU **80.84%**，作为项目验收与申报材料依据。

```bash
cd /workspace/segman/segmentation

python tools/test.py \
  local_configs/segman_trans/segman_b_trans10k_lass.py \
  --checkpoint outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --eval mIoU
```

**验收**：Summary 中 mIoU ≈ **80.84**；window ≈ **76.27**；bowl ≈ **80.07**。

---

### 步骤 ①-2：导出评测 json + 与基线对比表

**目的**：生成可归档的数值文件；论文/答辩直接引用 Δ 表。

```bash
python tools/test.py \
  local_configs/segman_trans/segman_b_trans10k_lass.py \
  --checkpoint outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --eval mIoU \
  --work-dir outputs/trans10k_lass_mmscope_fix5k/eval_final

python scripts/compare_miou_vs_baseline.py \
  "outputs/trans10k_lass_mmscope_fix5k/eval_final/eval_single_scale_*.json"
```

**验收**：终端打印 12 类 ↑/↓/≈；json 保存在 `eval_final/`。

---

### 步骤 ①-3：可视化（mask / 边界）

**目的**：软著、专利附图、报告「window / bowl 改善」目视证据。

```bash
python tools/test.py \
  local_configs/segman_trans/segman_b_trans10k_lass.py \
  --checkpoint outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --eval mIoU \
  --show-dir outputs/trans10k_lass_mmscope_fix5k/vis_deliver
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
| 分割权重 | `outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth` |
| 训练配置 | `local_configs/segman_trans/segman_b_trans10k_lass.py` |
| 对比文档 | `路线B_基线_fix5k_balanced10k_对比分析.md` |
| 数值来源 | `路线B_LASS_MMSCopE_实施清单.md` §0.1 |

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

**目的**：路线 C 只依赖该接口；内部固定加载 **fix5k** + `segman_b_trans10k_lass.py`。

**新建**：`transgrasp/segmentation/infer_segman.py`（接口见《项目实施步骤指南.md》B8）

**建议配置** `transgrasp/configs/segman_route_b.yaml`：

```yaml
config: local_configs/segman_trans/segman_b_trans10k_lass.py
checkpoint: outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth
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
    config='local_configs/segman_trans/segman_b_trans10k_lass.py',
    checkpoint='outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth',
)
img = cv2.imread('data/trans10k/images/val/<某张>.jpg')
out = p.predict(img)
assert out['sem_seg'].shape[:2] == img.shape[:2]
print('classes', out['sem_seg'].min(), out['sem_seg'].max())
"
```

**验收**：输出 `(H,W)` 语义图，类别 0～11；可选 `mask_union`（非背景并集）供抓取用。

---

### 步骤 ②-3：单张推理与基线/fix5k 目视对比（可选）

**目的**：抓取前确认 mask 边界可用。

```bash
# 使用 mmseg 自带推理或 ②-2 API，对同一张 val 图输出 sem_seg 上色图
# 与 outputs/trans10k_lass_mmscope_fix5k/vis_deliver 中 fix5k 结果一致即可
```

---

## 阶段 ③ 路线 C：抓取仿真 + UI（约 2～4 周）

> **前提**：阶段 ①、② 完成；**全程使用 fix5k 权重**，不再切换 balanced10k。

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

**验收**：给定 fix5k 导出的二值 mask，能返回合法 `4x4` 或 `pos+quat`。

---

### 步骤 ③-3：单次抓取闭环

**目的**：验证「分割 → 选实例 → 规划 → PyBullet 执行」端到端。

**新建**：`transgrasp/grasping/run_grasp_sim_once.py`

```bash
cd /workspace/segman
python transgrasp/grasping/run_grasp_sim_once.py \
  --image data/trans10k/images/val/<sample>.jpg \
  --checkpoint segmentation/outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth
```

**验收**：至少 1 个场景完成 approach → close → lift；失败有日志。

---

### 步骤 ③-4：抓取参数搜索（可选）

**目的**：提高仿真成功率；对应设计书 G0/G1。

**新建**：`transgrasp/grasping/optimize_grasp.py`

**验收**：同一物体上，优化后成功率 ≥ 默认参数。

---

### 步骤 ③-5：Gradio UI

**目的**：演示「上传图 → fix5k 分割 → 点选 → 仿真抓取」。

```bash
pip install gradio
python transgrasp/ui/app.py
```

**验收**：非开发者可完成全流程；UI 内写死 fix5k 路径或读 `segman_route_b.yaml`。

---

### 步骤 ③-6：端到端批量表（10～20 张）

**目的**：结题/论文「分割 + 抓取」联合指标。

| 列 | 内容 |
|----|------|
| 图像 id | val 文件名 |
| mIoU / 类 IoU | 与 GT 对比（可选） |
| mask 质量 | 目视 0/1 |
| 仿真成功 | 0/1 |

**验收**：表格可复现；README 记录命令与 fix5k 路径。

---

## 阶段 ④ 文档与申报收尾（与阶段 ③ 并行）

### 步骤 ④-1：更新项目主文档中的「正式权重」

**目的**：全文一致写 fix5k，避免读者误用 balanced10k。

| 文件 | 动作 |
|------|------|
| `项目实施步骤指南.md` | 路线 C 前提改为 fix5k 路径 |
| `路线B_LASS_MMSCopE_实施清单.md` §3.9 | 主推 fix5k（若仍为 balanced，改文案） |
| `Trans10K_SegMAN_B_训练与评测结果.md` | 增一节「路线 B fix5k 结果」指向 §0.1 |

---

### 步骤 ④-2：软著 / 专利材料

**目的**：申报材料与运行版本一致。

| 材料 | 建议内容 |
|------|----------|
| **软著** | 软件名 + V1.0；源码节选含 LASS/MMSCopE + `segman_b_trans10k_lass.py`；说明书用 **fix5k** 训练/推理流程 |
| **专利** | 技术方案：LASS + MMSCopE；**主实施例实验**：基线 vs **fix5k**（见对比分析文档）；balanced10k 仅作从属/可选实施例 |

---

### 步骤 ④-3：论文/答辩表格（固定一张主表）

**目的**：主文只报 fix5k vs 基线；balanced 放附录或消融。

**主表数据**：直接复制《路线B_基线_fix5k_balanced10k_对比分析.md》§2～§3 中 **基线 | fix5k** 两列。

**一句话贡献**：在 Trans10K 上，LASS+MMSCopE 短程微调达 mIoU **80.84%**，window IoU **+9.65%**，bowl **+1.16%**。

---

## 不再执行的项（已决策冻结）

| 项 | 原因 |
|----|------|
| balanced10k 继续训 / 部署 | bowl 回落；与 fix5k 交付目标冲突 |
| bowl5k、bowl3k_from10k 再训 | 未达 mIoU+bowl 双达标或已实验失败 |
| fix80k / 80k 无修复权重 | mIoU 塌缩，见实施清单 §0.2 |
| 从 iter_10000 再微调 bowl | 1b 三次 test 未达标，见平衡微调 §11 |

消融实验**保留在仓库与对比文档中**，写论文「补充实验」即可，**不进入 transgrasp 默认配置**。

---

## 命令速查（fix5k 唯一）

```bash
# 训练（已完成，仅备查；勿覆盖 fix5k 除非有意重训）
python tools/train.py local_configs/segman_trans/segman_b_trans10k_lass.py \
  --work-dir outputs/trans10k_lass_mmscope_fix5k \
  --load-from outputs/trans10k_segman_b/iter_80000.pth \
  --no-validate \
  --cfg-options runner.max_iters=5000 data.workers_per_gpu=2 optimizer.lr=3e-5

# 正式评测
python tools/test.py local_configs/segman_trans/segman_b_trans10k_lass.py \
  --checkpoint outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --eval mIoU

# 路线 C 开发时：所有推理指向上述 config + checkpoint
```

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-23 | 初版：项目后续以 fix5k 为唯一正式权重，分 ①～④ 阶段 |
