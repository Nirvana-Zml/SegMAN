# 路线 C：细分类（Grounded-SAM 特征）→ ASGrasp → PyBullet 仿真

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.1 |
| 日期 | 2026-05-23 |
| 前置（已完成） | 路线 B：**fix5k** `outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth` |
| 关联设计 | 《透明物体分割_SegMAN优化设计说明书.md》§6～§7；《项目实施步骤指南.md》路线 C |

---

## 1. 系统流水线（你要做的三件事）

```text
┌─────────────────────────────────────────────────────────────────┐
│ ① SegMAN（fix5k）语义分割 — 已完成                                │
│    输入：RGB 图  →  输出：12 类语义图 / 实例 mask / ROI bbox       │
└───────────────────────────────┬─────────────────────────────────┘
                                │ mask 裁剪 ROI
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ ② 细分类（可选）：Grounded-SAM 冻结特征 + 自研轻量分类头            │
│    输入：ROI 图像（+ 可选 SegMAN 特征）  →  输出：细粒度类别 + 置信度 │
└───────────────────────────────┬─────────────────────────────────┘
                                │ class_id + mask（+ 深度/点云若可用）
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ ③ 抓取：ASGrasp 位姿估计 → PyBullet 仿真执行与参数优化             │
│    输入：mask + 类别  →  输出：6D 抓取位姿 + 仿真成功/失败          │
└─────────────────────────────────────────────────────────────────┘
```

**职责边界（重要）**

| 模块 | 做什么 | 不做什么 |
|------|--------|----------|
| **SegMAN fix5k** | 像素级 **分割**、实例 **定位**（mask/bbox） | 一般不单独承担最终细分类（可选粗类来自 12 类语义图） |
| **Grounded-SAM** | 冻结 **视觉—语言/通用特征** | 一般不端到端训大模型（算力与稳定性） |
| **轻量分类头**（若实施） | 在 ROI 上 **训练/推理细分类** | 分割 |
| **ASGrasp** | **抓取位姿**（6-DoF） | 分割与分类 |
| **PyBullet** | **仿真验证**抓取、迭代夹爪/接近参数 | 真实机器人（可后续扩展） |

SegMAN 的 12 类（box、bottle、window…）可用于：**ROI 粗类别**、训练细分类时的弱标签、或 UI 默认类；若需更高精度细分类，可在 Grounded-SAM 特征上另行训练分类头。

---

## 2. 阶段 0：分割侧收尾（fix5k，约 1 天）

**目的**：路线 C 全程固定同一分割权重，避免 mask 漂移。

```bash
docker exec -it segman_train bash
conda activate segman
cd /workspace/segman/segmentation

python tools/test.py local_configs/segman_trans/segman_b_trans10k_lass.py \
  --checkpoint outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --eval mIoU
```

**交付**：`infer_segman` API（见《路线B_fix5k_项目后续步骤.md》阶段 ②）  
**输出接口**：`sem_seg (H,W)`、`instances[{mask, bbox, coarse_label}]`、`mask_union`（`label>0` 透明前景）

---

## 3. 阶段 ①：细分类数据（约 3～5 天）

### 步骤 ①-1：从 Trans10K 导出 ROI 数据集

**目的**：每张图、每个实例 → 裁剪图 + 细分类标签（Trans10K-v2 原有类别或你们定义的更细标签）。

**建议目录**：

```text
data/trans10k_roi/
├── train/
│   ├── images/          # 裁剪 ROI，如 cup_000123.jpg
│   └── labels.csv       # path, class_id, class_name, src_image, instance_id
├── val/
└── meta/
    └── classes.txt
```

**做法**：

1. 用 **GT mask**（训练分类头）或 **fix5k 预测 mask**（更贴近部署）做连通域；  
2. 按 bbox 外扩 10%～20% 裁剪；  
3. 标签 = 该实例在 GT 中的语义类（或映射到细分类体系）。

**脚本（待建）**：`transgrasp/data/build_roi_dataset.py`  
**输入**：`data/trans10k/` + 可选 fix5k 推理结果  
**验收**：每类样本数统计表；随机 20 张 ROI 目视无错位。

---

### 步骤 ①-2：划分 train/val、类均衡检查

**目的**：避免分类头只学会 background/大类。

**验收**：`labels.csv` 中每类 val ≥ 若干样本；长尾类记录增广策略（复制、轻微颜色抖动）。

---

## 4. 阶段 ②：大模型特征 + 细分类训练（约 1～2 周）

### 步骤 ②-1：环境 — Grounded-SAM（或 SAM2 + Grounding DINO）

**目的**：冻结特征提取器，不在此阶段训大模型。

```bash
# 示例：按所选仓库 README 安装（版本以论文/课题为准）
pip install segment-anything-grounding  # 或官方 Grounded-SAM / SAM2 组合
# 下载 checkpoint 到 transgrasp/checkpoints/grounded_sam/
```

**验收**：对单张 ROI 能 forward 得到特征向量（如 256～1024 维），无 OOM。

---

### 步骤 ②-2：实现特征提取封装

**目的**：统一接口，供细分类数据集 `__getitem__` 调用。

**新建**：`transgrasp/classification/grounded_sam_encoder.py`

```python
class GroundedSAMFeatureExtractor:
    def extract(self, image_bgr, mask=None) -> np.ndarray:
        """返回 F_fm，训练时可缓存到磁盘加速。"""
```

**可选**：离线预提取特征 → `data/trans10k_roi/features/train/*.npy`（强烈推荐，训练分类头更快）。

---

### 步骤 ②-3：实现轻量分类头并训练

**目的**：可训模块；输入 `Concat(F_grounded_sam, F_segman_roi可选)` → `K` 类 logits。

**新建**（命名自定，示例）：

```text
transgrasp/classification/roi_classifier.py
transgrasp/classification/train_classifier.py
transgrasp/configs/classifier.yaml
```

**训练策略（设计书 §6.4）**：

| 阶段 | 做法 |
|------|------|
| T1 | **冻结** SegMAN + Grounded-SAM，**只训** 轻量分类头（MLP / 小 Transformer） |
| T2（可选） | 小 lr 联合微调 ROI 上 SegMAN 特征分支 |

**命令示例**：

```bash
cd /workspace/segman
python transgrasp/classification/train_classifier.py \
  --config transgrasp/configs/classifier.yaml \
  --roi-root data/trans10k_roi \
  --work-dir outputs/classifier \
  --epochs 50 \
  --batch-size 32
```

**验收**：

- val **Top-1 Acc** / F1 达到课题要求（建议先 >80%，按类报表）；  
- 保存 `outputs/classifier/best.pth`；  
- 推理脚本对单 ROI < 100ms（GPU）。

---

### 步骤 ②-4：端到端「分割 + 细分类」联调

**目的**：验证 mask → crop → 分类 整条链。

**新建**：`transgrasp/pipelines/segment_and_classify.py`

```bash
python transgrasp/pipelines/segment_and_classify.py \
  --image data/trans10k/images/val/xxx.jpg \
  --seg-config segmentation/local_configs/segman_trans/segman_b_trans10k_lass.py \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --classifier-checkpoint outputs/classifier/best.pth
```

**验收**：输出每个实例的 `bbox, coarse_seg_class, fine_class, confidence`；低置信度可标「需人工确认」。

---

## 5. 阶段 ③：ASGrasp 抓取位姿（约 1～2 周）

### 步骤 ③-1：安装 ASGrasp 与权重

**目的**：输入 **RGB（+ 深度若需要）+ mask + 类别** → 6D grasp pose。

```bash
# 克隆 ASGrasp 官方/论文仓库至 third_party/ASGrasp 或 transgrasp/third_party/
# 按其 README 安装依赖、下载预训练权重
```

**新建**：`transgrasp/grasping/asgrasp_wrapper.py`

```python
def predict_grasp(image, mask, class_id=None) -> dict:
    # returns: position, quaternion, width, score
```

**验收**：对静态图 + mask 能返回合法位姿；score 可排序多个候选抓取。

---

### 步骤 ③-2：类别相关抓取先验

**目的**：不同透明类（cup / bottle / bowl）夹爪开度、接近距离不同。

**新建**：`transgrasp/configs/grasp_class_prior.yaml`

```yaml
cup:
  gripper_width: 0.06
  approach_offset: 0.08
bowl:
  gripper_width: 0.10
  ...
```

**验收**：ASGrasp 输出经 yaml 微调后，仿真成功率有提升（阶段 ④ 测）。

---

## 6. 阶段 ④：PyBullet 仿真抓取（约 1～2 周）

### 步骤 ④-1：仿真场景

**目的**：桌面 + 机械臂 + 夹爪 + 简单透明物体 URDF/网格。

**新建**：

```text
transgrasp/grasping/pybullet_env.py
transgrasp/configs/sim_scene.yaml
transgrasp/assets/   # URDF、mesh
```

```bash
pip install pybullet
python transgrasp/grasping/pybullet_env.py  # 冒烟：加载场景、关节可动
```

---

### 步骤 ④-2：单次抓取闭环

**目的**：**SegMAN mask → 类别 → ASGrasp 位姿 → PyBullet 执行**。

**新建**：`transgrasp/grasping/run_grasp_sim_once.py`

```bash
python transgrasp/grasping/run_grasp_sim_once.py \
  --image data/trans10k/images/val/xxx.jpg \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --classifier-checkpoint outputs/classifier/best.pth
```

**流程**：approach → close → lift → 判定是否抬起（成功/失败日志）。

**验收**：≥1 个场景完整跑通；失败时有明确原因（碰撞、滑脱、位姿无效）。

---

### 步骤 ④-3：抓取参数优化（可选）

**目的**：提高仿真成功率，对应设计书「仿真迭代」。

**新建**：`transgrasp/grasping/optimize_grasp.py`  
**方法**：对 `approach_dist, gripper_width, angle_offset` 网格/随机搜索，PyBullet 打分。

---

### 步骤 ④-4：Gradio 一体化 UI

**目的**：演示与答辩：上传图 → 分割叠加 → 点选实例 → 显示细分类 → 一键仿真。

```bash
pip install gradio
python transgrasp/ui/app.py
```

---

## 7. 阶段 ⑤：系统验收与文档（约 3～5 天）

| 交付物 | 内容 |
|--------|------|
| `transgrasp/README.md` | 环境、fix5k / 分类器 / ASGrasp 路径、一键命令 |
| 端到端表 | 10～20 张 val：分割 IoU（可选）、细分类 Acc、仿真成功率 |
| 论文/答辩图 | 原图 \| SegMAN mask \| 细分类标签 \| PyBullet 截图 |

---

## 8. 建议排期（参考）

| 周 | 任务 |
|----|------|
| 1 | 阶段 0 + ① ROI 数据集 |
| 2 | ② Grounded-SAM 特征 + 分类头训练 |
| 3 | ② 联调 + ③ ASGrasp |
| 4 | ④ PyBullet + UI + ⑤ 验收 |

---

## 9. 工程目录（目标结构）

```text
SegMAN/
├── segmentation/                    # 已有；fix5k 权重在此
├── transgrasp/
│   ├── segmentation/infer_segman.py
│   ├── classification/
│   │   ├── grounded_sam_encoder.py
│   │   ├── roi_classifier.py
│   │   └── train_classifier.py
│   ├── grasping/
│   │   ├── asgrasp_wrapper.py
│   │   ├── pybullet_env.py
│   │   ├── run_grasp_sim_once.py
│   │   └── optimize_grasp.py
│   ├── pipelines/
│   │   ├── segment_and_classify.py
│   │   └── run_interactive.py
│   ├── ui/app.py
│   └── configs/
│       ├── segman_route_b.yaml      # fix5k 路径
│       ├── classifier.yaml
│       ├── grasp_class_prior.yaml
│       └── sim_scene.yaml
├── data/trans10k_roi/               # 细分类 ROI
└── outputs/
    ├── trans10k_lass_mmscope_fix5k/  # 分割
    └── classifier/                   # 细分类（若实施）
```

---

## 10. 与路线 B 的衔接（fix5k 固定）

| 环节 | 使用 fix5k 的方式 |
|------|-------------------|
| 训练 ROI 集（可选） | 用 fix5k 预测 mask 裁 ROI，更贴近部署 |
| 训练分类头 | 建议 GT mask 裁 ROI（标签干净）；部署用 fix5k mask |
| ASGrasp / PyBullet | **必须** fix5k（或同一 `infer_segman`）出 mask |
| 论文叙述 | 阶段二 SegMAN+LASS+MMSCopE；阶段三 Grounded-SAM 特征 + 抓取仿真 |

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-23 | 初版：细分类 → ASGrasp → PyBullet 全链路步骤 |
| 2026-05-23 | v1.1：细分类改为可选通用表述 |
