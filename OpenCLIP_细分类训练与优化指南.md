# OpenCLIP 透明物体 ROI 细分类 — 训练与优化指南


| 项目          | 内容                                                                                   |
| ----------- | ------------------------------------------------------------------------------------ |
| 文档版本 | **v1.2** |
| **项目根目录** | `D:\SegMAN-main\SegMAN`（conda 环境 `segman`） |
| **OpenCLIP 源码** | `D:\SegMAN-main\open_clip-3`（与 SegMAN **平级**，不在 SegMAN 内） |
| 编写日期        | 2026-05-24（v1.2 更新 2026-05-25）                                                           |
| 适用场景        | SegMAN **v2@6k** 分割 → ROI 裁剪 → **OpenCLIP 视觉编码器 + 分类头** → 11 类细分类                    |
| 前置分割        | `segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth`（mIoU **81.80%**）    |
| 分割评测归档    | `.../eval_deliver_6k/eval_single_scale_20260524_133106.json`                                 |
| 分割配置        | `local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py`                   |
| OpenCLIP 安装 | 在 `segman` 环境中 `pip install open_clip_torch`（推荐）；或 `cd D:\SegMAN-main\open_clip-3` 后 `pip install -e .` |
| 关联文档 | 《路线C_细分类与抓取实施步骤.md》《..\open_clip-3\OpenCLIP_代码结构与SegMAN细分类应用分析.md》 |


---

## 0. 目录与虚拟环境约定（必读）

```text
D:\SegMAN-main\
├── SegMAN\                    ← 项目根：conda segman、segmentation、transgrasp
└── open_clip-3\               ← OpenCLIP 源码（兄弟目录，非 SegMAN 子目录）
```

| 用途 | 工作目录 |
|------|----------|
| OpenCLIP 细分类 | `D:\SegMAN-main\SegMAN` |
| SegMAN 分割 | `D:\SegMAN-main\SegMAN\segmentation` |
| editable 安装 OpenCLIP | `D:\SegMAN-main\open_clip-3` |

Docker 若只挂载 SegMAN → `/workspace/segman`，容器内**默认没有** open_clip-3；请用 PyPI 包或额外挂载 `open_clip-3`。

**MMSeg 数据路径（Docker / 本地均已转换）**：

| 用途 | 路径 |
|------|------|
| MMSeg Trans10K（推荐 `--data-root`） | `segmentation/data/trans10k/` |
| 项目根下（若存在） | `data/trans10k/` |
| HF 原始 | `data/Trans10K-v2/` |

---

## 1. 目标与原则

### 1.1 要做什么

在 Trans10K **11 类前景**透明物体上，对已裁剪的 ROI 小图做 **有监督细分类**：

`background` 不参与 ROI 训练；前景类与 `segmentation/local_configs/_base_/datasets/trans10k.py` 中 `CLASSES[1:]` 一致：

`box, bottle, window, eyeglass, freezer, jar_kettle, door, cup, wall, bowl, shelf`

### 1.2 不做什么


| 操作                                           | 是否采用  | 原因                              |
| -------------------------------------------- | ----- | ------------------------------- |
| `python -m open_clip_train.main` + LAION 图文对 | **否** | 解决的是 CLIP 预训练 scaling，不是闭集 11 类 |
| 用 OpenCLIP 做语义分割                             | **否** | 分割由 SegMAN 负责                   |
| 纯零样本文本 prompt 作生产分类器                         | **否** | 易混类（cup/bowl/jar）在透明域上不稳定       |
| 端到端反传更新 SegMAN                               | **否** | 分割权重固定 v2@6k                    |


### 1.3 训练策略总览

```text
阶段 0  环境 + 冒烟
阶段 1  ROI 数据（GT 训练 / SegMAN 评测）
阶段 2  零样本基线（可选，用于对比）
阶段 3  T1：冻结 OpenCLIP ViT + 训练 Linear/MLP 头  ← 主方案
阶段 4  T2：解冻 ViT 末 1～2 block + 小 lr（可选优化）
阶段 5  部署评测 + 端到端联调
阶段 6  超参与结构优化（按需迭代）
```

**核心公式**：`logits = Head( OpenCLIP.encode_image( preprocess(roi) ) )`，损失为 **CrossEntropy**（可加 label smoothing、class weights）。

### 1.4 当前进度与 T1 基线（2026-05-25）

| 阶段 | 状态 | 说明 |
|------|------|------|
| §2 环境 + 冒烟 | ✅ | ViT-B-16 feature dim 512 |
| §3-1～3-4 数据 | ✅ | 见下表 |
| §4 分类代码 | ✅ | `transgrasp/classification/` 已建 |
| §6 T1 linear | ✅ | 首轮 baseline 70.18% |
| §8 sweep | ✅ | **best 72.62%**（无 class_weights） |
| §6 Z0 零样本 | ⬜ | `eval_openclip_zeroshot.py` 待建 |
| §7-3 E2E | ⬜ | `segment_and_classify.py` 待建 |

**ROI 数据集（已导出）**：

| 目录 | ROI 数 | 用途 |
|------|--------|------|
| `data/trans10k_roi_gt/train` | 15,746 | T1 训练 |
| `data/trans10k_roi_gt/val` | 3,105 | GT 上限评测 |
| `data/trans10k_roi_segman/val` | 3,233 | 部署向评测 |

**§8 Sweep 最优**（`outputs/openclip_classifier/sweep_s8/s8_lr1e3_noweight/best.pth`，2026-05-25）：

| 评测集 | Top-1 Acc | macro F1 | vs baseline |
|--------|-----------|----------|-------------|
| GT-ROI val | **72.62%** | 70.43% | **+2.44 pt** |
| SegMAN-ROI val | **64.12%** | 58.45% | **+1.67 pt** |
| **ΔAcc** | **−8.50%** | — | — |

**§8 五组排名（GT-ROI Acc）**：

| 排名 | 实验 | Acc | 结论 |
|------|------|-----|------|
| 1 | **无 class_weights**（lr=1e-3 linear） | **72.62%** | **采用** |
| 2 | MLP lr=5e-4 | 70.24% | 无显著增益 |
| 3 | baseline（有 weights） | 70.18% | 被替代 |
| 4 | label_smoothing=0 | 70.11% | 无增益 |
| 5 | lr=5e-4 linear | 70.02% | 无增益 |
| 6 | lr=1e-2 linear | 68.63% | 过大，勿用 |

**各类 F1（§8 best，GT-ROI）**：eyeglass 94%、cup 91%、freezer 83% 提升明显；**shelf 49%**、window 51%、door 57% 仍为瓶颈。

**结论**：§8 P1～P4 已完成；**关闭 class_weights** 为唯一有效改动；距 **≥80%** 仍差 ~7 pt，下一步 **P6 ViT-L-14** 或 **P7 T2 解冻**。

**T1 首轮（已被 §8 best 替代，保留作对照）**：

| 评测集 | Top-1 Acc | macro F1 | 说明 |
|--------|-----------|----------|------|
| GT-ROI val | **70.18%** | 67.67% | 训练上限 |
| SegMAN-ROI val | **62.45%** | 57.90% | 部署向 |
| **ΔAcc** | **−7.73%** | — | 分割误差传递 |

**各类 F1（T1 首轮）**：eyeglass 93%、cup 90% 较好；shelf 41%、window 50%、door 55%。

---

## 2. 环境与依赖

### 步骤 2-1：进入训练环境

**目的**：在 **SegMAN 项目根** 激活已有 conda 环境；OpenCLIP 与 SegMAN 共用该环境。

**Windows（当前推荐）**：

```powershell
conda activate segman
cd D:\SegMAN-main\SegMAN
```

**Docker（可选）**：仅当 SegMAN 目录挂载为 `/workspace/segman` 时：

```bash
docker exec -it segman_train bash
source /root/anaconda3/etc/profile.d/conda.sh && conda activate segman
cd /workspace/segman
# open_clip-3 不在此目录内；请用 pip install open_clip_torch
```

---

### 步骤 2-2：安装 OpenCLIP

**目的**：在 **segman** 环境中安装 `import open_clip`；加载 ViT 与官方 `preprocess`。

**方式 A — PyPI（推荐，最简单）**：

在 `D:\SegMAN-main\SegMAN` 下、已激活 `segman` 后执行：

```powershell
pip install -U open_clip_torch timm huggingface-hub safetensors
```

**方式 B — 锁定版本**：

```powershell
pip install open_clip_torch==2.32.0 "timm>=1.0.17"
```

**方式 C — 本地源码 editable（需改 open_clip 代码时）**：

源码在 **兄弟目录** `D:\SegMAN-main\open_clip-3`，**不是** `SegMAN\open_clip-3`：

```powershell
conda activate segman
cd D:\SegMAN-main\open_clip-3
pip install -e .
cd D:\SegMAN-main\SegMAN
python -c "import open_clip; print(open_clip.__file__)"
```

**验收**：最后一行应指向 `site-packages\open_clip\`（方式 A/B）或 `open_clip-3\src\open_clip\`（方式 C）。

**不要**在 SegMAN 目录内执行 `pip install -e .`（SegMAN 根目录不是 OpenCLIP 包）。

---

### 步骤 2-3：冒烟测试

**目的**：确认 GPU、权重下载、`encode_image` 输出维度正常。

```powershell
cd D:\SegMAN-main\SegMAN

python -c "
import open_clip, torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained='laion2b_s34b_b88k')
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
x = preprocess(Image.new('RGB', (224, 224))).unsqueeze(0)
if torch.cuda.is_available():
    x = x.cuda()
with torch.no_grad():
    f = model.encode_image(x)
print('feature dim:', f.shape[-1])
print('cuda:', torch.cuda.is_available())
"
```

**验收**：打印 `feature dim: 512`（ViT-B-16），无报错、无 OOM。

---

### 步骤 2-4：列出可用预训练权重

**目的**：选型或更换 backbone 时查阅官方 tag。

```bash
python -c "
import open_clip
for m, p in open_clip.list_pretrained()[:20]:
    print(m, p)
print('... total', len(open_clip.list_pretrained()))
"
```

---

## 3. 数据准备

### 3.1 目录约定

**目的**：训练与评测分离 mask 来源，避免用 SegMAN 错 mask 污染标签。


| 目录                          | mask 来源         | 用途                   |
| --------------------------- | --------------- | -------------------- |
| `data/trans10k_roi_gt/`     | Trans10K **GT** | **训练**分类头、T1/T2 主验证  |
| `data/trans10k_roi_segman/` | **v2@6k** 预测    | **部署向**评测、E2E、误差传递分析 |


```text
data/trans10k_roi_gt/
├── train/images/          # train_cup_000123.jpg
├── train/labels.csv
├── val/images/
├── val/labels.csv
└── meta/
    ├── classes.txt        # 11 行前景类名
    └── class_weights.npy  # 可选，长尾类加权

data/trans10k_roi_segman/  # 结构同上
```

`labels.csv` 建议列：`path,class_id,class_name,src_image,instance_id,mask_source`

---

### 步骤 3-1：生成类别表

**目的**：与 SegMAN `CLASSES` 下标对齐，推理写回时用同一 id。

```bash
cd D:\SegMAN-main\SegMAN
mkdir -p data/trans10k_roi_gt/meta

python -c "
classes = '''background box bottle window eyeglass freezer jar_kettle door cup wall bowl shelf'''.split()
open('data/trans10k_roi_gt/meta/classes.txt','w',encoding='utf-8').write('\n'.join(classes[1:]))
print('wrote', len(classes)-1, 'foreground classes')
"
```

**验收**：`classes.txt` 共 11 行，顺序与 `trans10k.py` 中 `CLASSES[1:]` 一致。

---

### 步骤 3-2：从 GT mask 导出训练 ROI

**目的**：得到标签干净的 crop 图，作为 OpenCLIP **主训练集**。

**脚本**：`transgrasp/data/build_roi_dataset.py`（**已完成**）

```bash
cd D:\SegMAN-main\SegMAN   # Docker: cd /workspace/segman

python transgrasp/data/build_roi_dataset.py \
  --data-root segmentation/data/trans10k \
  --split train \
  --mask-source gt \
  --out-root data/trans10k_roi_gt/train \
  --bbox-pad 0.15 \
  --min-area 64

python transgrasp/data/build_roi_dataset.py \
  --data-root segmentation/data/trans10k \
  --split val \
  --mask-source gt \
  --out-root data/trans10k_roi_gt/val \
  --bbox-pad 0.15 \
  --min-area 64
```


| 参数                 | 目的                          |
| ------------------ | --------------------------- |
| `--mask-source gt` | 用真值 mask，crop 与 class_id 可靠 |
| `--bbox-pad 0.15`  | 外扩 15% 保留透明边缘与上下文           |
| `--min-area 64`    | 过滤过小实例，减少噪声 ROI             |


**验收**：

```bash
python transgrasp/data/stats_roi_dataset.py --root data/trans10k_roi_gt
```

每类 train 样本数建议 ≥ 50；val 每类 ≥ 30（**freezer** val 仅 20，属数据集长尾）。随机目视 20 张 crop，类名与图像一致。

**已验收（2026-05-24）**：train 15,746 / val 3,105 ROI。

---

### 步骤 3-3：从 SegMAN v2@6k 导出部署向 ROI

**目的**：模拟上线「先分割再分类」，量化分割误差对 Acc 的影响。

**步骤 A — 保存 v2@6k 语义预测（class-id PNG，非可视化叠加图）**

> 注意：`tools/test.py --show-dir` 保存的是 RGB 叠加可视化，**不能**用于 ROI 裁剪。请用下方脚本导出单通道 label map（0–11）。

```bash
cd D:\SegMAN-main\SegMAN

python transgrasp/data/export_sem_seg_preds.py \
  --config segmentation/local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py \
  --checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --data-root segmentation/data/trans10k \
  --split val \
  --out-dir segmentation/outputs/trans10k_lass_mmscope_balanced_v2/pred_sem_seg_val
```

**验收**：`pred_sem_seg_val/` 下约 1000 张 PNG，文件名与 `img_dir/val/*.jpg` stem 一致（如 `val_000000.png`）。

**步骤 B — 由预测 mask 裁 ROI**

```bash
cd D:\SegMAN-main\SegMAN

python transgrasp/data/build_roi_dataset.py \
  --data-root segmentation/data/trans10k \
  --split val \
  --mask-source segman \
  --pred-dir segmentation/outputs/trans10k_lass_mmscope_balanced_v2/pred_sem_seg_val \
  --out-root data/trans10k_roi_segman/val \
  --bbox-pad 0.15 \
  --min-area 64
```

**验收**：`trans10k_roi_segman/val` 样本数与 GT-ROI 同量级；记录与 GT-ROI 的类别分布差异。

**已验收（2026-05-25）**：`pred_sem_seg_val/` 1000 张 class-id PNG；`trans10k_roi_segman/val` 3,233 ROI。

> **`pred_sem_seg_val/` 为何发黑？**  
> 保存的是 **单通道类别 id（0–11）**，不是彩色可视化。肉眼几乎全黑属正常；供 `build_roi_dataset.py` 读取，勿用 `tools/test.py --show-dir` 的 RGB 叠加图替代。

> **`eval_deliver_6k/` 与 `pred_sem_seg_val/` 区别**  
> - `eval_deliver_6k/`：mIoU 评测 JSON（`--work-dir`），**不含**预测 mask。  
> - `pred_sem_seg_val/`：§3-3 专用 class-id 语义图。

---

### 步骤 3-4：导出类别权重

**目的**：缓解 shelf/cup/bowl 等长尾或易混类不平衡。

```bash
python transgrasp/data/export_class_weights.py \
  --root data/trans10k_roi_gt \
  --split train \
  --export-weights data/trans10k_roi_gt/meta/class_weights.npy
```

**目的说明**：默认 `1/sqrt(freq)`，均值归一化为 1；可选 `--method inv|effective`。

**已验收（2026-05-25）**：`meta/class_weights.npy`（11 维 float32）。

---

## 4. 代码结构（已实现）

**目的**：训练、评测、推理共用同一 encoder + head，避免 preprocess 不一致。

```text
transgrasp/
├── configs/
│   └── openclip_classifier.yaml
├── classification/
│   ├── __init__.py              # 空文件即可，标记 Python 包
│   ├── openclip_encoder.py
│   ├── roi_classifier.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── checkpoint_utils.py
│   ├── train_openclip_classifier.py
│   ├── eval_openclip_classifier.py
│   └── infer_openclip_roi.py
├── data/
│   ├── build_roi_dataset.py
│   ├── export_sem_seg_preds.py  # §3-3：导出 class-id 预测 mask
│   ├── stats_roi_dataset.py
│   └── export_class_weights.py
└── pipelines/
    └── segment_and_classify.py    # 待建（§7-3）
```

**配置文件示例** `transgrasp/configs/openclip_classifier.yaml`：

```yaml
clip_model: ViT-B-16
clip_pretrained: laion2b_s34b_b88k
num_classes: 11
freeze_clip: true
head: linear              # linear | mlp
mlp_hidden: 256           # head=mlp 时
roi_root: data/trans10k_roi_gt
work_dir: outputs/openclip_classifier
label_smoothing: 0.1
```

---

## 5. 预训练模型选型

**目的**：在精度与显存之间权衡；默认 ViT-B-16 足够 T1，易混类不足时再升档。


| 阶段                | `--clip-model` | `--clip-pretrained` | 特征维 | 目的                           |
| ----------------- | -------------- | ------------------- | --- | ---------------------------- |
| **默认 T1**         | `ViT-B-16`     | `laion2b_s34b_b88k` | 512 | 速度与精度平衡，单卡友好                 |
| 精度优化              | `ViT-L-14`     | `laion2b_s32b_b82k` | 768 | T1 Acc 不足且显存 ≥ 12GB          |
| 轻量对照              | `ViT-B-32`     | `laion2b_s34b_b79k` | 512 | 快速实验、消融                      |
| 零样本对照 FROM OpenAI | `ViT-B-16`     | `openai`            | 512 | 与 LAION 权重对比（需 quickgelu 注意） |


> **注意**：OpenAI 原版权重需 `ViT-B-16` 且部分环境要 `force_quick_gelu=True`；LAION 权重用默认 GELU 即可。详见 `D:\SegMAN-main\open_clip-3\README.md` § Pretrained models。

---

## 6. 训练阶段与命令

### 阶段 Z0 — 零样本基线（可选）

**目的**：建立下界；论文/答辩可报「有监督头 vs 零样本」提升量。不用于部署。

```bash
cd D:\SegMAN-main\SegMAN

python transgrasp/classification/eval_openclip_zeroshot.py \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --clip-model ViT-B-16 \
  --clip-pretrained laion2b_s34b_b88k \
  --report-dir outputs/openclip_classifier/zeroshot_baseline
```

**实现要点**（若脚本未建，可临时用以下逻辑）：

- 用 `open_clip.build_zero_shot_classifier(model, tokenizer, classnames, templates)`；
- 类名 prompt 示例：`"a photo of a {class}."` 或 `"a transparent {class}."`；
- 对 ROI 做 `preprocess` → `encode_image(normalize=True)` → 与类嵌入点积 argmax。

**验收**：输出 `zeroshot_baseline/summary.json`（Acc、混淆矩阵）。预期 Acc **低于** T1 有监督头。

---

### 阶段 T1 — 冻结 ViT + 训练分类头（主方案）

**目的**：参数最少、收敛最快；对应 Scaling Laws 论文 **linear probing** 范式。

```bash
cd D:\SegMAN-main\SegMAN   # Docker: cd /workspace/segman

python transgrasp/classification/train_openclip_classifier.py \
  --config transgrasp/configs/openclip_classifier.yaml \
  --roi-root data/trans10k_roi_gt \
  --work-dir outputs/openclip_classifier/t1_freeze_vitb16 \
  --clip-model ViT-B-16 \
  --clip-pretrained laion2b_s34b_b88k \
  --freeze-clip \
  --head linear \
  --epochs 40 \
  --batch-size 64 \
  --lr 1e-3 \
  --weight-decay 0.01 \
  --label-smoothing 0.1 \
  --num-workers 4 \
  --class-weights data/trans10k_roi_gt/meta/class_weights.npy \
  --patience 8 \
  --seed 42
```

> **CLI 与 yaml**：命令行显式传入的参数（如 `--epochs`）**优先于** yaml 默认值；避免冒烟时 yaml 的 `epochs: 40` 覆盖 `--epochs 1`。


| 参数                      | 推荐值       | 目的                  |
| ----------------------- | --------- | ------------------- |
| `--freeze-clip`         | 开启        | 只更新分类头，保护 CLIP 通用表征 |
| `--head linear`         | linear    | 512→11，参数量 ~5.6K    |
| `--lr 1e-3`             | 1e-3～1e-2 | 头参数量小，可用较大 lr       |
| `--epochs 40`           | 30～50     | 配合 early stop 防过拟合  |
| `--batch-size 64`       | 32～128    | 视显存调整               |
| `--label-smoothing 0.1` | 0～0.1     | 缓解易混类过自信            |
| `--class-weights`       | `.npy`    | 长尾类（shelf 等）加权      |
| `--patience 8`          | 5～10      | val 无提升时 early stop   |


**训练过程监控**：

```bash
# 训练日志在终端；结束后见 history.json
cat outputs/openclip_classifier/t1_freeze_vitb16/history.json

# 当前 best 指标（训练中实时更新）
cat outputs/openclip_classifier/t1_freeze_vitb16/eval_gt_roi/summary.json
```

**首轮实测（2026-05-25）**：

- 约 **48 min** / 16 epoch（early stop）；best @ **epoch 8**
- train loss 1.74 → 1.40 稳定下降；val Acc 在 **~69–70%** 平台期
- 每 epoch 约 **2–4.5 min**（首轮含模型下载更慢）

**验收**：

- 产出 `best.pth`、`last.pth`、`history.json`；
- GT-ROI val **Top-1 Acc ≥ 80%**（课题基线；**首轮 70.18% 未达标**）；
- `per_class_report.json` 中 cup/bowl/jar_kettle F1 可接受；
- train/val loss 曲线无严重发散。

**T1 评测**：

```bash
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint outputs/openclip_classifier/t1_freeze_vitb16/best.pth \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --report-dir outputs/openclip_classifier/t1_freeze_vitb16/eval_gt_roi
```

---

### 阶段 T1+ — MLP 头（结构优化，可选）

**目的**：线性头在易混类上不足时，增加一层非线性，仍保持 ViT 冻结。

```bash
python transgrasp/classification/train_openclip_classifier.py \
  --config transgrasp/configs/openclip_classifier.yaml \
  --roi-root data/trans10k_roi_gt \
  --work-dir outputs/openclip_classifier/t1_mlp_vitb16 \
  --clip-model ViT-B-16 \
  --clip-pretrained laion2b_s34b_b88k \
  --freeze-clip \
  --head mlp \
  --mlp-hidden 256 \
  --mlp-dropout 0.1 \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-4 \
  --weight-decay 0.01 \
  --class-weights data/trans10k_roi_gt/meta/class_weights.npy
```


| 参数                  | 目的                        |
| ------------------- | ------------------------- |
| `--head mlp`        | `512 → 256 → 11`，提升非线性可分性 |
| `--lr 5e-4`         | MLP 参数量更大，lr 略低于 linear   |
| `--mlp-dropout 0.1` | 减轻小数据集过拟合                 |


**验收**：同 GT-ROI val 上 Acc/F1 **不低于** linear T1，否则回退 linear。

---

### 阶段 T2 — 解冻 ViT 末层（精度优化，可选）

**目的**：T1 在 cup/bowl/bottle 等类 F1 仍低时，小幅微调视觉塔尾部。参考 WiSE-FT / Scaling Laws fine-tuning 思路，**小 lr、短 epoch**。

**前置**：T1 已收敛，`--resume` 加载 `t1_.../best.pth`。

```bash
python transgrasp/classification/train_openclip_classifier.py \
  --config transgrasp/configs/openclip_classifier.yaml \
  --roi-root data/trans10k_roi_gt \
  --work-dir outputs/openclip_classifier/t2_unfreeze2_vitb16 \
  --resume outputs/openclip_classifier/t1_freeze_vitb16/best.pth \
  --clip-model ViT-B-16 \
  --clip-pretrained laion2b_s34b_b88k \
  --freeze-clip false \
  --unfreeze-last-blocks 2 \
  --epochs 15 \
  --batch-size 32 \
  --lr 5e-6 \
  --head-lr 1e-4 \
  --weight-decay 0.05 \
  --label-smoothing 0.1 \
  --class-weights data/trans10k_roi_gt/meta/class_weights.npy
```


| 参数                         | 目的                                                               |
| -------------------------- | ---------------------------------------------------------------- |
| `--unfreeze-last-blocks 2` | 对应 `model.lock_image_tower(unlocked_groups=1)` 解冻 ViT 最后 block 组 |
| `--lr 5e-6`                | 视觉塔极小 lr，防破坏预训练                                                  |
| `--head-lr 1e-4`           | 分类头可用更大 lr                                                       |
| `--batch-size 32`          | 解冻后显存上升，适当减小                                                     |
| `--epochs 15`              | 短程微调，防过拟合                                                        |


**验收**：

- GT-ROI val Acc **≥ T1**；
- 若 Acc 或 macro-F1 下降 → **回退 T1**，不部署 T2；
- 记录 T1 vs T2 混淆矩阵差异（易混类是否改善）。

---

### 阶段 T3 — 更大 backbone（可选）

**目的**：T2 仍不足且 GPU 显存充足（≥ 16GB）时，换 ViT-L-14 重做 T1。

```bash
python transgrasp/classification/train_openclip_classifier.py \
  --roi-root data/trans10k_roi_gt \
  --work-dir outputs/openclip_classifier/t1_freeze_vitl14 \
  --clip-model ViT-L-14 \
  --clip-pretrained laion2b_s32b_b82k \
  --freeze-clip \
  --head linear \
  --epochs 40 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 0.01 \
  --class-weights data/trans10k_roi_gt/meta/class_weights.npy
```

**目的说明**：768 维特征通常线性可分性更好；代价是推理更慢、显存更高。

---

## 7. 部署向评测与端到端

### 步骤 7-1：SegMAN mask ROI 上评测分类头

**目的**：衡量 **分割 + 分类** 真实误差，与 GT-ROI 训练上限对比。

```bash
python transgrasp/classification/eval_openclip_classifier.py \
  --checkpoint outputs/openclip_classifier/t1_freeze_vitb16/best.pth \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir outputs/openclip_classifier/eval_on_segman_roi
```

**验收**：生成 `summary.json`（Acc、混淆矩阵）。记录：

```text
ΔAcc = Acc(GT-ROI) − Acc(SegMAN-ROI)   # 首轮：70.18% − 62.45% = 7.73%
```

用于说明 SegMAN v2@6k 分割瓶颈（尤其 shelf/box/door IoU 较低类）。

**已验收（2026-05-25）**：`outputs/openclip_classifier/eval_on_segman_roi/summary.json`。

---

### 步骤 7-2：单张 ROI 推理调试

**目的**：不跑 SegMAN，快速验证分类头与 preprocess。

```bash
python transgrasp/classification/infer_openclip_roi.py \
  --checkpoint outputs/openclip_classifier/t1_freeze_vitb16/best.pth \
  --image data/trans10k_roi_gt/val/images/val_cup_000000.jpg \
  --topk 3
```

**说明**：`best.pth` 内 `meta` 已含 `clip_model` / `clip_pretrained`，无需重复传 CLIP 参数。

**预期输出**：`fine_class=cup, confidence=0.91, top3=[...]`

---

### 步骤 7-3：端到端分割 + 细分类

**目的**：上线形态；输出驱动 ASGrasp。

```bash
cd D:\SegMAN-main\SegMAN

python transgrasp/pipelines/segment_and_classify.py \
  --backend openclip \
  --image data/trans10k/img_dir/val/xxxx.jpg \
  --seg-config segmentation/local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --classifier-checkpoint outputs/openclip_classifier/t1_freeze_vitb16/best.pth \
  --clip-model ViT-B-16 \
  --clip-pretrained laion2b_s34b_b88k \
  --bbox-pad 0.15 \
  --conf-threshold 0.5 \
  --out-dir outputs/demo_segment_classify
```


| 参数                     | 目的                     |
| ---------------------- | ---------------------- |
| `--conf-threshold 0.5` | 低于阈值标记「需人工确认」，避免错类驱动抓取 |
| `--bbox-pad 0.15`      | 与 ROI 数据集导出一致          |


**批量 val**：

```bash
python transgrasp/pipelines/segment_and_classify.py \
  --backend openclip \
  --image-list data/trans10k/meta/val_list.txt \
  --seg-config segmentation/local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --classifier-checkpoint outputs/openclip_classifier/t1_freeze_vitb16/best.pth \
  --out-json outputs/openclip_classifier/e2e_val_predictions.json
```

---

## 8. 超参数优化指南

**目的**：T1 未达标时，按优先级逐项调整，避免一次改太多无法归因。

> **首轮 T1 启示**：epoch 8 后 val 不再提升 → 优先调 **lr / head 结构**，而非单纯加 epoch；shelf/window/door 为弱类，与 SegMAN 分割弱项一致。

### 8.1 推荐搜索顺序


| 优先级 | 调什么                     | 搜索范围                     | 目的                                                 |
| --- | ----------------------- | ------------------------ | -------------------------------------------------- |
| 1   | 学习率 `--lr`              | `{1e-2, 1e-3, 5e-4}`     | 线性头最敏感（参考 Scaling Laws §4.3：{0.1, 0.01, 0.001} 量级） |
| 2   | `--head`                | linear → mlp             | 易混类非线性边界                                           |
| 3   | `--class-weights`       | 开/关；手动加重 shelf           | 分割弱类 ROI 噪声大                                       |
| 4   | `--label-smoothing`     | `{0, 0.05, 0.1}`         | 过自信误分类                                             |
| 5   | `--epochs` + early stop | 30～50，**patience=8**    | 首轮 epoch 8 即 best，过长训练无益 |
| 6   | backbone                | B-16 → L-14              | 特征容量                                               |
| 7   | T2 解冻                   | blocks=1～2, lr=1e-6～5e-5 | 最后手段                                               |


### 8.2 数据增强（谨慎）

**目的**：ROI 数较少时略增多样性；**必须在 OpenCLIP preprocess 之前**对 PIL 做轻量变换。


| 增强          | 建议                        | 目的            |
| ----------- | ------------------------- | ------------- |
| 水平翻转        | p=0.5                     | 透明物体左右对称类多    |
| ColorJitter | brightness/contrast ≤ 0.1 | 模拟反光，幅度宜小     |
| 禁止          | 大幅旋转、强模糊                  | 破坏 CLIP 预训练分布 |


训练脚本中应使用 `create_model_and_transforms` 返回的 `**preprocess_train`**（含 RandomResizedCrop），与 `**preprocess_val**` 区分。

### 8.3 易混类专项

**目的**：针对 cup / bowl / jar_kettle / bottle 混淆。

1. 导出混淆矩阵，定位 top 混淆对；
2. 提高 `--class-weights` 中对应类；
3. 检查 ROI 是否含大量背景（SegMAN shelf/box 分割 IoU 低时尤甚）；
4. 尝试 T2 或 ViT-L-14；
5. E2E 部署提高 `--conf-threshold`，低置信度走人工确认。

### 8.4 预处理一致性（必检）

**目的**：train/val/infer 必须使用同一套 OpenCLIP normalize。

```python
# 正确：全程用 factory 返回的 preprocess
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(...)
# 训练用 preprocess_train；验证/推理用 preprocess_val
```

### 8.5 批量执行 §8 搜索（推荐）

**脚本**：`transgrasp/classification/run_sweep_s8.py` — 按 §8.1 优先级顺序训练并写入 `sweep_results.json`。

```bash
cd /workspace/segman   # 或 D:\SegMAN-main\SegMAN

# 全流程（约 4～6 h：5 组实验 × ~1h）
python transgrasp/classification/run_sweep_s8.py

# 仅跑 P1 学习率
python transgrasp/classification/run_sweep_s8.py --ids s8_lr1e2_linear s8_lr5e4_linear

# 仅跑 P2 MLP（在 P1 最优 lr 确定后）
python transgrasp/classification/run_sweep_s8.py --ids s8_mlp5e4
```

**输出**：

| 文件 | 说明 |
|------|------|
| `outputs/openclip_classifier/sweep_s8/sweep_results.json` | 各实验 val Acc |
| `outputs/openclip_classifier/sweep_s8/sweep_summary.json` | 与 baseline 排序 |
| `outputs/openclip_classifier/sweep_s8/<exp_id>/best.pth` | 各组权重 |

**实验列表**（相对 baseline `t1_freeze_vitb16` @ 70.18%）：

| id | §8 优先级 | 变更 |
|----|----------|------|
| `s8_lr1e2_linear` | P1 | lr=1e-2 |
| `s8_lr5e4_linear` | P1 | lr=5e-4 |
| `s8_mlp5e4` | P2 | MLP + lr=5e-4 |
| `s8_lr1e3_noweight` | P3 | 关闭 class_weights |
| `s8_lr1e3_ls0` | P4 | label_smoothing=0 |

Sweep 结束后取 `sweep_summary.json` 中 **best**，再跑 §7-1 SegMAN-ROI 评测；若仍 <80%，再考虑 §8.1 的 P6 ViT-L-14 或 P7 T2。

---


| 路径 | 说明 | 状态 |
|------|------|------|
| `outputs/openclip_classifier/sweep_s8/s8_lr1e3_noweight/best.pth` | **§8 当前最优** | ✅ 72.62% / 64.12% |
| `outputs/openclip_classifier/t1_freeze_vitb16/best.pth` | T1 首轮 baseline | ✅ 70.18% |
| `.../eval_gt_roi/summary.json` | GT-ROI Acc / 混淆矩阵 | ✅ |
| `outputs/openclip_classifier/eval_on_segman_roi/summary.json` | SegMAN-ROI 部署向 | ✅ 62.45% |
| `.../per_class_report.json` | 每类 P/R/F1 | ✅ |
| `segmentation/outputs/.../pred_sem_seg_val/` | v2@6k class-id mask | ✅ |
| `.../e2e_val_predictions.json` | 端到端预测 | ⬜ |
| `outputs/openclip_classifier/zeroshot_baseline/` | 零样本对照 | ⬜ |


**best.pth 内容**（已实现）：

```python
{
  'head': {...},           # ClassificationHead state_dict
  'meta': {
    'clip_model': 'ViT-B-16',
    'clip_pretrained': 'laion2b_s34b_b88k',
    'head': 'linear',
    'num_classes': 11,
    'feat_dim': 512,
    'class_names': [...],
  },
  'epoch': 8,
  'val_acc': 0.7018,
}
```

---

## 10. 常见问题


| 现象                               | 可能原因             | 处理                                                 |
| -------------------------------- | ---------------- | -------------------------------------------------- |
| `pred_sem_seg_val/` 图片全黑         | class-id PNG 非可视化 | 正常；目视请用 ROI crop 或彩色 overlay |
| `--show-dir` 不能裁 ROI             | 输出 RGB 叠加图       | 改用 `export_sem_seg_preds.py` |
| `eval_deliver_6k/` 无预测 mask      | 仅 mIoU JSON       | §3-3 需单独导出 `pred_sem_seg_val/` |
| yaml 覆盖 CLI `--epochs`           | apply_config 顺序   | 已修复：CLI 优先；或不用 `--config` |
| val Acc ~70% 达不到 80%            | 线性头 + 易混类      | §8：MLP / 调 lr / T2 |
| `feature dim` 报错 / Unknown model | timm 过旧          | `pip install -U timm`                              |
| HF 权重下载失败                        | 网络               | 镜像或手动下载 `.bin` 后 `--clip-pretrained /path/to/ckpt` |
| train Acc 高、val Acc 低            | 过拟合              | 减 epoch、加 dropout、label smoothing                  |
| GT-ROI 高、SegMAN-ROI 低            | 分割瓶颈             | 固定 v2@6k；报告 ΔAcc；优先改善 mask 弱类                      |
| cup/bowl 互混                      | 类间相似 + ROI 背景    | class weights、T2、缩小 bbox_pad                       |
| OOM                              | batch 过大 / T2 解冻 | 减 batch、梯度累积、仅 T1                                  |
| 误跑 `open_clip_train`             | 理解偏差             | 停止；改用本节 T1 脚本                                      |


---

## 11. 与路线 C / ASGrasp 的衔接

```text
SegMAN v2@6k (iter_6000.pth)
        ↓
OpenCLIP T1 best.pth  ──→  segment_and_classify.py
        ↓
class_id + mask + confidence
        ↓
ASGrasp predict_grasp()  ──→  PyBullet 仿真
```

- 分割权重：**仅** `trans10k_lass_mmscope_balanced_v2/iter_6000.pth`
- 分类权重：本节 T1 `best.pth`（或验收通过的 T2）
- 抓取先验：`transgrasp/configs/grasp_class_prior.yaml` 按 `fine_class_name` 映射

---

## 12. 命令速查

**Windows**（`D:\SegMAN-main\SegMAN`）：

```powershell
conda activate segman
cd D:\SegMAN-main\SegMAN

# 数据 §3-2
python transgrasp/data/build_roi_dataset.py --data-root segmentation/data/trans10k --split train --mask-source gt --out-root data/trans10k_roi_gt/train --bbox-pad 0.15 --min-area 64

# 数据 §3-4
python transgrasp/data/export_class_weights.py --root data/trans10k_roi_gt --split train --export-weights data/trans10k_roi_gt/meta/class_weights.npy

# T1 训练 §6
python transgrasp/classification/train_openclip_classifier.py `
  --roi-root data/trans10k_roi_gt --work-dir outputs/openclip_classifier/t1_freeze_vitb16 `
  --clip-model ViT-B-16 --clip-pretrained laion2b_s34b_b88k `
  --freeze-clip --head linear --epochs 40 --batch-size 64 --lr 1e-3 `
  --class-weights data/trans10k_roi_gt/meta/class_weights.npy --patience 8

# GT-ROI 评测
python transgrasp/classification/eval_openclip_classifier.py `
  --checkpoint outputs/openclip_classifier/t1_freeze_vitb16/best.pth `
  --roi-root data/trans10k_roi_gt --split val `
  --report-dir outputs/openclip_classifier/t1_freeze_vitb16/eval_gt_roi

# SegMAN-ROI 评测 §7-1
python transgrasp/classification/eval_openclip_classifier.py `
  --checkpoint outputs/openclip_classifier/t1_freeze_vitb16/best.pth `
  --roi-root data/trans10k_roi_segman --split val `
  --report-dir outputs/openclip_classifier/eval_on_segman_roi
```

**Docker**（`/workspace/segman`）：将上述路径中的 `D:\SegMAN-main\SegMAN` 换为 `/workspace/segman`，命令相同。

**分割 mIoU 归档**（与 §3-3 预测 mask **无关**）：

```powershell
cd D:\SegMAN-main\SegMAN\segmentation
python tools/test.py local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py `
  --checkpoint outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth `
  --eval mIoU --work-dir outputs/trans10k_lass_mmscope_balanced_v2/eval_deliver_6k
```

---

## 修订记录


| 日期         | 说明                                      |
| ---------- | --------------------------------------- |
| 2026-05-24 | v1.0：OpenCLIP ROI 细分类训练、优化、评测、E2E 步骤与命令 |
| 2026-05-24 | v1.1：路径对齐 Windows 本地布局（SegMAN 与 open_clip-3 平级） |
| 2026-05-25 | v1.2：§3～§4 实测落地；T1 基线 70.18%/62.45%；修正 data-root、`export_sem_seg_preds`、FAQ 与产出物状态 |
| 2026-05-25 | v1.2.1：§8.5 增加 `run_sweep_s8.py` 批量调参流程 |
| 2026-05-25 | v1.2.2：§8 sweep 完成；best=无 class_weights 72.62% GT / 64.12% SegMAN |


