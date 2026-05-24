# 路线 C：细分类（OpenCLIP / 可选 Grounded-SAM）→ ASGrasp → PyBullet 仿真

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.2 |
| 日期 | 2026-05-23 |
| 前置（已完成） | 路线 B：**fix5k** `outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth` |
| 细分类推荐方案 | **OpenCLIP** 预训练视觉编码器 + 轻量分类头（**不做**语义分割、**不**跑官方 CLIP 大规模图文对比预训练） |
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
│ ② 细分类（推荐）：OpenCLIP 视觉编码器 + 线性/MLP 分类头              │
│    输入：SegMAN mask 裁出的透明物体 ROI  →  输出：12 类细类 + 置信度   │
│    （可选备选：Grounded-SAM 冻结特征 + 同类分类头，见 §4.7）          │
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
| **SegMAN fix5k** | 像素级 **分割**、实例 **定位**（mask/bbox） | **细分类**（交给 OpenCLIP 头） |
| **OpenCLIP** | 在 ROI 上做 **有监督细分类**（冻结或小幅微调 ViT） | **语义分割**、大规模 image–text 对比预训练 |
| **轻量分类头** | `Linear` / 小 MLP：`encode_image(roi)` → 11 类 logits | 分割 |
| **Grounded-SAM**（可选） | 替代 OpenCLIP 作 ROI 特征 | 端到端训大模型 |
| **ASGrasp** | **抓取位姿**（6-DoF） | 分割与分类 |
| **PyBullet** | **仿真验证**抓取、迭代夹爪/接近参数 | 真实机器人（可后续扩展） |

**细分类标签空间**（与 Trans10K / SegMAN 一致，共 **12 类含 background**；ROI 训练通常 **11 类前景**，`class_id` 与 `segmentation/local_configs/_base_/datasets/trans10k.py` 中 `CLASSES` 下标对齐）：

`background`, `box`, `bottle`, `window`, `eyeglass`, `freezer`, `jar_kettle`, `door`, `cup`, `wall`, `bowl`, `shelf`

> **说明**：OpenCLIP 只消费 **SegMAN 已分割并裁好的透明物体小图**；分割质量由 fix5k 决定，分类器不反向训练 SegMAN（除非另开 T2 实验）。

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

## 3. 阶段 ①：ROI 数据（SegMAN / GT mask → 裁剪图，约 2～4 天）

细分类训练数据 = **透明物体实例 ROI 图 + 类别 id**。mask 来源分两套（用途不同）：

| 数据集后缀 | mask 来源 | 目的 |
|------------|-----------|------|
| `trans10k_roi_gt/` | Trans10K **GT** 标注 | **训练** OpenCLIP 分类头（标签干净、上限高） |
| `trans10k_roi_segman/` | **fix5k 预测** mask | **验证 / 部署评测**（误差传递与真实流水线一致） |

### 步骤 ①-1：目录与 `classes.txt`

**目的**：统一路径与类别表，供导出脚本、DataLoader、推理结果写回共用。

```text
data/trans10k_roi_gt/
├── train/images/              # 如 train_box_000123.jpg
├── train/labels.csv
├── val/images/
├── val/labels.csv
└── meta/classes.txt           # 11 行前景类名（无 background）或 12 行含 background

data/trans10k_roi_segman/      # 结构同上，由 fix5k 推理生成
```

`labels.csv` 列建议：`path,class_id,class_name,src_image,instance_id,mask_source`

**命令（生成 meta，一次性）**：

```bash
docker exec -it segman_train bash
conda activate segman
cd /workspace/segman

mkdir -p data/trans10k_roi_gt/meta
python -c "
classes = '''background box bottle window eyeglass freezer jar_kettle door cup wall bowl shelf'''.split()
open('data/trans10k_roi_gt/meta/classes.txt','w').write('\n'.join(classes[1:]))
print('wrote', len(classes)-1, 'foreground classes')
"
```

**验收**：`classes.txt` 与 `trans10k.py` 中 `CLASSES[1:]` 顺序一致。

---

### 步骤 ①-2：用 GT mask 导出训练 ROI（推荐先做）

**目的**：从全图语义标注得到「每个连通域一张 crop + 细类标签」，作为 OpenCLIP **主训练集**。

**脚本（待建）**：`transgrasp/data/build_roi_dataset.py`

```bash
cd /workspace/segman

python transgrasp/data/build_roi_dataset.py \
  --data-root data/trans10k \
  --split train \
  --mask-source gt \
  --out-root data/trans10k_roi_gt/train \
  --bbox-pad 0.15 \
  --min-area 64

python transgrasp/data/build_roi_dataset.py \
  --data-root data/trans10k \
  --split val \
  --mask-source gt \
  --out-root data/trans10k_roi_gt/val \
  --bbox-pad 0.15 \
  --min-area 64
```

| 参数 | 目的 |
|------|------|
| `--mask-source gt` | 用 `ann_dir` 真值 mask，避免 SegMAN 错裁污染训练标签 |
| `--bbox-pad 0.15` | bbox 外扩 15%，保留透明物体边缘与少量上下文 |
| `--min-area 64` | 过滤过小连通域，减少噪声 ROI |

**验收**：`python transgrasp/data/stats_roi_dataset.py --root data/trans10k_roi_gt` 输出每类样本数；随机 20 张 ROI 目视无错位、类名与 crop 一致。

---

### 步骤 ①-3：用 SegMAN（fix5k）导出部署向 ROI

**目的**：模拟上线流程——**先分割再裁图**——用于 val 评测与困难样本分析（允许与 GT-ROI 准确率对比）。

**步骤 A — 批量保存 fix5k 语义图（若尚无）**

```bash
cd /workspace/segman/segmentation

python tools/test.py local_configs/segman_trans/segman_b_trans10k_lass.py \
  --checkpoint outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --eval mIoU \
  --show-dir outputs/trans10k_lass_mmscope_fix5k/pred_sem_seg_val
```

**目的**：得到与 val 集对齐的预测 mask，供 ROI 裁剪；`--show-dir` 下 PNG 与 `data/trans10k/img_dir/val` 同名对应（以你们 `test.py` 实际输出为准，必要时在 `build_roi_dataset.py` 中指定 `--pred-dir`）。

**步骤 B — 由预测 mask 裁 ROI**

```bash
cd /workspace/segman

python transgrasp/data/build_roi_dataset.py \
  --data-root data/trans10k \
  --split val \
  --mask-source segman \
  --pred-dir segmentation/outputs/trans10k_lass_mmscope_fix5k/pred_sem_seg_val \
  --seg-config segmentation/local_configs/segman_trans/segman_b_trans10k_lass.py \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --out-root data/trans10k_roi_segman/val \
  --bbox-pad 0.15
```

**目的**：评估「SegMAN 裁错 / 漏实例」对细分类 Acc 的影响；**不用于** OpenCLIP 主训练（除非做联合鲁棒性实验）。

**验收**：`trans10k_roi_segman/val/labels.csv` 行数与实例数合理；与 GT-ROI 相比记录 `crop_iou` 或「标签是否因 mask 错误而噪声」比例。

---

### 步骤 ①-4：类均衡与增广策略

**目的**：避免分类头只学会 `bottle`、`cup` 等大类。

```bash
python transgrasp/data/stats_roi_dataset.py \
  --root data/trans10k_roi_gt \
  --split train \
  --export-weights data/trans10k_roi_gt/meta/class_weights.npy
```

**做法**：长尾类在 `train_classifier` 中使用 `--class-weights`；或复制 ROI + 轻微 `ColorJitter`（在 OpenCLIP `preprocess` 之外对 PIL 图做，幅度宜小）。

**验收**：val 集每类样本数 ≥ 课题下限（建议每类 ≥30）；`class_weights.npy` 已生成。

---

## 4. 阶段 ②：OpenCLIP 细分类训练与执行（约 3～7 天）

### 步骤总览

| 步骤 | 做什么 | 目的 |
|------|--------|------|
| ②-1 | `pip install open_clip_torch` + 冒烟 | 确认预训练 ViT 可加载 |
| ②-2 | 实现 `openclip_encoder` / `train_openclip_classifier.py` | 统一训练与推理代码 |
| ②-3 | **T1** 冻结 CLIP，只训线性头 | 主方案：ROI 有监督 11 类细分类 |
| ②-4 | **T2**（可选）解冻 ViT 末层 | 提升易混类，小 lr 微调 |
| ②-5 | 在 `trans10k_roi_segman` 上 eval | 衡量 SegMAN+分类 真实精度 |
| ②-6 | `infer_openclip_roi.py` 单张 crop | 调试分类头 |
| ②-7 | `segment_and_classify.py --backend openclip` | **执行**：原图 → 分割 → 细类 |
| ②-8 | — | 避免误用官方 CLIP 大规模预训练 |
| ②-9 | Grounded-SAM（可选） | 特征对比实验，非默认 |

> **原则（必读）**  
> - **不**使用 `python -m open_clip_train.main` 做 LAION 式大规模图文对比预训练。  
> - **只**在 ROI 上做 **有监督分类**（CrossEntropy）：`OpenCLIP.encode_image(roi)` → 分类头 → 11 类。  
> - SegMAN 权重全程 **冻结**；OpenCLIP 默认 **冻结 ViT，只训头**（阶段 T1），可选 T2 解冻 ViT 最后 1～2 个 block。

### 步骤 ②-1：安装 OpenCLIP 推理/训练依赖

**目的**：加载预训练 ViT 与官方 `preprocess`（resize、normalize 与预训练一致，否则特征失真）。

```bash
docker exec -it segman_train bash
conda activate segman
cd /workspace/segman

pip install -U open_clip_torch timm
# 可选：固定版本便于复现
# pip install open_clip_torch==2.32.0 timm>=1.0.15
```

**冒烟（确认 GPU 与权重可加载）**：

```bash
python -c "
import open_clip, torch
from PIL import Image
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained='laion2b_s34b_b88k')
model.eval()
x = preprocess(Image.new('RGB', (224, 224))).unsqueeze(0)
with torch.no_grad():
    f = model.encode_image(x)
print('feature dim', f.shape[-1])
"
```

**目的**：确认环境、CUDA、HuggingFace/timm 权重下载正常。  
**验收**：打印 `feature dim`（ViT-B-16 多为 512），无 OOM。

**推荐预训练权重（按算力选）**：

| `--clip-model` | `--clip-pretrained` | 目的 |
|----------------|---------------------|------|
| `ViT-B-16` | `laion2b_s34b_b88k` | 默认：速度与精度平衡 |
| `ViT-L-14` | `laion2b_s32b_b82k` | 更高精度，显存更大 |

---

### 步骤 ②-2：实现 OpenCLIP ROI 分类器（待建脚本）

**目的**：封装「ROI 图像 → 类别 logits」，训练与推理共用同一模块。

**新建文件**：

```text
transgrasp/classification/openclip_encoder.py    # 加载 OpenCLIP，encode_image
transgrasp/classification/roi_classifier.py        # Linear / MLP 头
transgrasp/classification/train_openclip_classifier.py
transgrasp/classification/eval_openclip_classifier.py
transgrasp/configs/openclip_classifier.yaml
```

**配置示例** `transgrasp/configs/openclip_classifier.yaml`：

```yaml
clip_model: ViT-B-16
clip_pretrained: laion2b_s34b_b88k
num_classes: 11          # 前景类；不含 background
freeze_clip: true        # T1
head: linear             # linear | mlp
roi_root: data/trans10k_roi_gt
work_dir: outputs/openclip_classifier
```

---

### 步骤 ②-3：阶段 T1 — 冻结 OpenCLIP，只训分类头

**目的**：用最少参数、最快收敛，在 Trans10K ROI 上学会 11 类细分类；避免破坏 CLIP 预训练表征。

```bash
cd /workspace/segman

python transgrasp/classification/train_openclip_classifier.py \
  --config transgrasp/configs/openclip_classifier.yaml \
  --roi-root data/trans10k_roi_gt \
  --work-dir outputs/openclip_classifier/t1_freeze \
  --clip-model ViT-B-16 \
  --clip-pretrained laion2b_s34b_b88k \
  --freeze-clip \
  --head linear \
  --epochs 40 \
  --batch-size 64 \
  --lr 1e-3 \
  --weight-decay 0.01 \
  --num-workers 4 \
  --class-weights data/trans10k_roi_gt/meta/class_weights.npy
```

| 参数 | 目的 |
|------|------|
| `--freeze-clip` | 仅更新 `Linear(512→11)`，显存小、训练稳 |
| `--lr 1e-3` | 分类头常用较大 lr |
| `--class-weights` | 缓解 cup/bowl/jar 等长尾类不平衡 |
| `--roi-root ..._gt` | 用 GT 裁的 ROI，标签可靠 |

**验收**：

- `outputs/openclip_classifier/t1_freeze/best.pth` 存在；  
- 在 `data/trans10k_roi_gt/val` 上 **Top-1 Acc ≥ 80%**（课题可再调高）；  
- `outputs/openclip_classifier/t1_freeze/per_class_report.json` 含每类 P/R/F1。

---

### 步骤 ②-4：阶段 T2（可选）— 解冻 ViT 最后若干 block

**目的**：当 T1 在易混类（`cup`/`bowl`/`jar_kettle`）上 F1 偏低时，用小 lr 微调视觉编码器末尾层。

```bash
python transgrasp/classification/train_openclip_classifier.py \
  --config transgrasp/configs/openclip_classifier.yaml \
  --roi-root data/trans10k_roi_gt \
  --work-dir outputs/openclip_classifier/t2_unfreeze \
  --resume outputs/openclip_classifier/t1_freeze/best.pth \
  --freeze-clip false \
  --unfreeze-last-blocks 2 \
  --epochs 15 \
  --batch-size 32 \
  --lr 5e-6 \
  --head-lr 1e-4 \
  --weight-decay 0.05
```

| 参数 | 目的 |
|------|------|
| `--resume t1/best.pth` | 在已收敛分类头基础上继续 |
| `--unfreeze-last-blocks 2` | 只动 ViT 尾部，降低灾难性遗忘 |
| `--lr 5e-6` | CLIP 视觉塔用小 lr（参考 WiSE-FT 思路） |

**验收**：T2 在 **GT-ROI val** 上 Acc 不低于 T1；若下降则回退只用 T1。

---

### 步骤 ②-5：在 SegMAN 预测 mask 裁的 ROI 上评测

**目的**：衡量真实流水线（分割误差 + 分类）的表现，与训练集（GT-ROI）区分。

```bash
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint outputs/openclip_classifier/t1_freeze/best.pth \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir outputs/openclip_classifier/eval_on_segman_roi
```

**验收**：生成 `eval_on_segman_roi/summary.json`（Acc、混淆矩阵）；记录相对 GT-ROI eval 的 **Acc 落差**（用于论文/答辩说明 SegMAN 瓶颈）。

---

### 步骤 ②-6：单张 ROI 推理（仅细分类）

**目的**：不跑 SegMAN，只对已有 crop 图做分类（调试头、标注工具）。

```bash
python transgrasp/classification/infer_openclip_roi.py \
  --checkpoint outputs/openclip_classifier/t1_freeze/best.pth \
  --image data/trans10k_roi_gt/val/images/cup_001042.jpg \
  --topk 3
```

**预期输出示例**：`fine_class=cup, confidence=0.91, top3=[...]`

---

### 步骤 ②-7：端到端「SegMAN 分割 + OpenCLIP 细分类」

**目的**：上线形态——上传原图 → fix5k mask → 每实例 crop → OpenCLIP 细类。

**脚本（待建）**：`transgrasp/pipelines/segment_and_classify.py`

```bash
cd /workspace/segman

python transgrasp/pipelines/segment_and_classify.py \
  --backend openclip \
  --image data/trans10k/img_dir/val/xxxx.jpg \
  --seg-config segmentation/local_configs/segman_trans/segman_b_trans10k_lass.py \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --classifier-checkpoint outputs/openclip_classifier/t1_freeze/best.pth \
  --clip-model ViT-B-16 \
  --clip-pretrained laion2b_s34b_b88k \
  --bbox-pad 0.15 \
  --conf-threshold 0.5 \
  --out-dir outputs/demo_segment_classify
```

| 参数 | 目的 |
|------|------|
| `--backend openclip` | 使用 OpenCLIP 路径（与可选 `grounded_sam` 区分） |
| `--bbox-pad` | 与 ROI 数据集导出一致 |
| `--conf-threshold` | 低于阈值标「需人工确认」，避免错类驱动抓取 |

**批量 val 评测**：

```bash
python transgrasp/pipelines/segment_and_classify.py \
  --backend openclip \
  --image-list data/trans10k/meta/val_list.txt \
  --seg-config segmentation/local_configs/segman_trans/segman_b_trans10k_lass.py \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --classifier-checkpoint outputs/openclip_classifier/t1_freeze/best.pth \
  --out-json outputs/openclip_classifier/e2e_val_predictions.json
```

**验收**：每张图输出 `instances[]`：`bbox, seg_class_id, fine_class_id, fine_class_name, confidence`；JSON 可驱动阶段 ③ ASGrasp。

---

### 步骤 ②-8：训练与推理对照（避免误用）

| 操作 | 是否推荐 | 说明 |
|------|----------|------|
| `open_clip_train.main` + 海量 caption | **否** | 与 SegMAN ROI 细分类无关 |
| 文本 prompt 零样本（不写训练脚本） | 仅作 **基线对比** | 透明物体上通常弱于有监督头 |
| 在 ROI 上 `CrossEntropy` + 冻结/微解冻 ViT | **是** | 本路线标准做法 |
| 用 OpenCLIP 做语义分割 | **否** | 属于另一条技术路线 |

---

### 步骤 ②-9（可选备选）：Grounded-SAM 特征 + 同类分类头

**目的**：若课题要求对比「通用 grounding 特征」与 OpenCLIP，可并行训练；**默认以 OpenCLIP 为准**。

```bash
pip install segment-anything-grounding   # 按所选仓库 README
# checkpoint → transgrasp/checkpoints/grounded_sam/

python transgrasp/classification/train_openclip_classifier.py \
  --feature-backend grounded_sam \
  --roi-root data/trans10k_roi_gt \
  --work-dir outputs/grounded_sam_classifier/t1_freeze \
  --epochs 40 --batch-size 32
```

**验收**：与 OpenCLIP T1 在相同 `trans10k_roi_gt/val` 上对比 Acc/F1；权重目录勿与 `outputs/openclip_classifier/` 混用。

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
  --image data/trans10k/img_dir/val/xxx.jpg \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --classifier-checkpoint outputs/openclip_classifier/t1_freeze/best.pth \
  --classifier-backend openclip
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
| `transgrasp/README.md` | 环境、fix5k / **OpenCLIP 分类器** / ASGrasp 路径、一键命令 |
| 端到端表 | 10～20 张 val：GT-ROI Acc、SegMAN-ROI Acc、E2E 细分类 Acc、仿真成功率 |
| 论文/答辩图 | 原图 \| SegMAN mask \| OpenCLIP 细类标签 \| PyBullet 截图 |

---

## 8. 建议排期（参考）

| 周 | 任务 |
|----|------|
| 1 | 阶段 0 + ① GT/SegMAN 双路 ROI 导出 |
| 2 | ② OpenCLIP T1 训练 + GT-ROI / SegMAN-ROI 评测 |
| 3 | ② E2E 联调（`segment_and_classify`）+ ③ ASGrasp |
| 4 | ④ PyBullet + UI + ⑤ 验收；（可选 T2 微调、Grounded-SAM 对比） |

---

## 9. 工程目录（目标结构）

```text
SegMAN/
├── segmentation/                         # 已有；fix5k 权重在此
├── transgrasp/
│   ├── data/
│   │   ├── build_roi_dataset.py          # GT / SegMAN mask → ROI
│   │   └── stats_roi_dataset.py
│   ├── segmentation/infer_segman.py
│   ├── classification/
│   │   ├── openclip_encoder.py           # OpenCLIP 封装（推荐）
│   │   ├── roi_classifier.py
│   │   ├── train_openclip_classifier.py
│   │   ├── eval_openclip_classifier.py
│   │   ├── infer_openclip_roi.py
│   │   └── grounded_sam_encoder.py       # 可选对比
│   ├── grasping/ ...
│   ├── pipelines/segment_and_classify.py
│   └── configs/
│       ├── segman_route_b.yaml
│       ├── openclip_classifier.yaml      # OpenCLIP 细分类
│       └── grasp_class_prior.yaml
├── data/
│   ├── trans10k/                         # 原始数据
│   ├── trans10k_roi_gt/                  # GT mask 裁 ROI（训练）
│   └── trans10k_roi_segman/              # fix5k 裁 ROI（部署评测）
└── outputs/
    ├── trans10k_lass_mmscope_fix5k/      # 分割
    └── openclip_classifier/              # 细分类权重与评测
        ├── t1_freeze/best.pth
        └── eval_on_segman_roi/
```

---

## 10. 与路线 B 的衔接（fix5k 固定）

| 环节 | 使用 fix5k 的方式 |
|------|-------------------|
| OpenCLIP **训练** ROI | **GT mask**（`trans10k_roi_gt`） |
| OpenCLIP **部署评测** ROI / E2E | **fix5k mask**（`trans10k_roi_segman` + `segment_and_classify`） |
| ASGrasp / PyBullet | **必须** fix5k（或同一 `infer_segman`）出 mask |
| 论文叙述 | 阶段二 SegMAN 分割；阶段三 **OpenCLIP ROI 细分类** + 抓取仿真 |

**OpenCLIP 一键命令速查**（容器内 `/workspace/segman`）：

```bash
# 1) 环境
pip install -U open_clip_torch timm

# 2) 导出 GT 训练 ROI
python transgrasp/data/build_roi_dataset.py --mask-source gt --split train ...

# 3) 训练（T1 冻结 CLIP）
python transgrasp/classification/train_openclip_classifier.py \
  --roi-root data/trans10k_roi_gt --work-dir outputs/openclip_classifier/t1_freeze --freeze-clip

# 4) E2E：SegMAN + OpenCLIP
python transgrasp/pipelines/segment_and_classify.py --backend openclip \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth \
  --classifier-checkpoint outputs/openclip_classifier/t1_freeze/best.pth
```

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-23 | 初版：细分类 → ASGrasp → PyBullet 全链路步骤 |
| 2026-05-23 | v1.1：细分类改为可选通用表述 |
| 2026-05-23 | v1.2：**OpenCLIP** 根据 SegMAN 透明语义分割 ROI 做细分类；补充安装/导出/训练/评测/E2E 命令与目的；Grounded-SAM 降为可选对比 |
