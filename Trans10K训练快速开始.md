# Trans10K-v2 直接训练 SegMAN（快速开始）

> 不依赖 ADE20K。在 Docker/WSL 中路径以 `/workspace/segman` 为例，Windows 请改为 `D:\SegMAN-main\SegMAN`。

---

## 训练目的与在本课题中的角色

本仓库在 Trans10K-v2 上训练 SegMAN，**首要目标是学好透明物体的像素级分割（mask / ROI）**，为后续模块提供空间定位。**最终细粒度类别不由 SegMAN 输出**。

| 模块 | 职责 |
|------|------|
| **SegMAN**（本文） | 透明物体 **在哪儿**：输出分割 mask，供裁剪 ROI |
| **Grounded-SAM**（路线 C） | 在 mask 区域内提取大模型特征、参与语义理解 |
| **TransFine**（路线 C） | 融合 Grounded-SAM + SegMAN ROI 特征，做 **细分类** |
| **抓取仿真**（路线 C） | 在 mask + 类别基础上做 PyBullet / ASGrasp 等 |

```text
RGB 图像
   → SegMAN：透明区域 mask（本阶段产物）
   → 按 mask 裁剪 ROI
   → Grounded-SAM + TransFine：细分类（非 SegMAN 完成）
   → 抓取 / 交互系统
```

### 为何配置里仍是 12 类？

Trans10K-v2 官方标注为 **12 类语义分割**（背景 + 11 类透明物体）。当前默认 `segman_b_trans10k.py` 使用 `num_classes=12`，含义是：

- **训练 / 评测**：用多类标注学习更清晰的边界与形状，并用 **mIoU** 与论文/基线对比；
- **系统集成**：下游一般只取 **前景 mask**（或将多类预测合并为「透明物体」区域），**不把 SegMAN 的 class id 当作最终分类结果**。

若更贴近「只分割、不分类」的部署目标，可使用 **二分类**（背景 vs 透明），见第 3 步与 `segman_b_trans10k_binary.py`。

更完整的系统架构见《透明物体分割_SegMAN优化设计说明书.md》《项目实施步骤指南.md》路线 B～C。

---

## 第 1 步：拉全 HuggingFace 数据（必做）

你本地的 `data/Trans10K-v2/` 若只有 `README.md`，说明 **LFS 大文件还没下完**。

```bash
cd /workspace/segman/data/Trans10K-v2
git lfs install
git lfs pull
ls data/    # 应看到 train-*.parquet、validation-*.parquet
```

---

## 第 2 步：安装转换依赖

```bash
conda activate segman
pip install datasets pyarrow
```

---

## 第 3 步：导出为 MMSeg 格式

在 **`segmentation`** 目录下执行：

```bash
cd /workspace/segman/segmentation

# 全量（train + val）
python tools/convert_datasets/trans10k.py \
  ../data/Trans10K-v2 \
  -o data/trans10k \
  --mode multiclass

# 调试：每个 split 只导 20 张
# python tools/convert_datasets/trans10k.py ../data/Trans10K-v2 -o data/trans10k --max-samples 20
```

验收：

```bash
ls data/trans10k/img_dir/train | head
ls data/trans10k/ann_dir/train | head
```

应成对出现 `train_000000.jpg` 与 `train_000000.png`。

**二分类（更贴近下游「只出 mask」）**：背景 vs 任意透明区域，`num_classes=2`，与「分类由 Grounded-SAM + TransFine 完成」一致；使用配置 `segman_b_trans10k_binary.py`：

```bash
python tools/convert_datasets/trans10k.py ../data/Trans10K-v2 -o data/trans10k --mode binary
```

多类与二分类二选一即可；**当前 80k 正式训练若已用 multiclass，无需为重训而改**，系统集成时合并前景类即可。

---

## 第 4 步：检查训练配置

预训练权重路径（已写在配置里，按你机器改一行即可）：

`segmentation/local_configs/segman_trans/segman_b_trans10k.py`

```python
pretrained='../pretrained/SegMAN_Encoder_b.pth.tar',
```

Docker 常用：

```python
pretrained='/workspace/segman/pretrained/SegMAN_Encoder_b.pth.tar',
```

打印配置：

```bash
cd /workspace/segman/segmentation
python tools/print_config.py local_configs/segman_trans/segman_b_trans10k.py
```

确认 `num_classes=12`（或二分类时为 `2`）、`data_root='data/trans10k'`。  
`num_classes=12` **不代表**部署时用 SegMAN 做最终物体分类，见上文「训练目的与在本课题中的角色」。  
数据集配置使用 MMSeg **v0.30** 的 `img_dir` / `ann_dir`（不要用 `data_prefix`）。

---

## 第 5 步：训练

### 冒烟（约 20 iter）

```bash
python tools/train.py local_configs/segman_trans/segman_b_trans10k.py \
  --work-dir outputs/trans10k_smoke \
  --cfg-options runner.max_iters=20 data.samples_per_gpu=1 data.workers_per_gpu=2
```

### Debug 短训（2000 iter，先看 mIoU 趋势）

```bash
python tools/train.py local_configs/segman_trans/segman_b_trans10k_debug.py \
  --work-dir outputs/trans10k_debug
```

- 每 **500 iter** 在验证集评估并更新 `best_mIoU*.pth`
- 日志：`outputs/trans10k_debug/*.log.json`
- 满意后再跑下面 80k 正式训练

### 正式单卡（80000 iter）

```bash
python tools/train.py local_configs/segman_trans/segman_b_trans10k.py \
  --work-dir outputs/trans10k_segman_b \
  --cfg-options data.workers_per_gpu=2 evaluation.interval=8000
```

验证阶段若出现 **`Killed`**（多为容器内存 OOM），可改为训练时不验证、训完再测：

```bash
python tools/train.py local_configs/segman_trans/segman_b_trans10k.py \
  --work-dir outputs/trans10k_segman_b \
  --resume-from outputs/trans10k_segman_b/iter_16000.pth \
  --no-validate \
  --cfg-options data.workers_per_gpu=2
```

### 从 checkpoint 续训

```bash
python tools/train.py local_configs/segman_trans/segman_b_trans10k.py \
  --work-dir outputs/trans10k_segman_b \
  --resume-from outputs/trans10k_segman_b/iter_16000.pth \
  --no-validate \
  --cfg-options data.workers_per_gpu=2
```

`latest.pth` 若已指向最新 iter，也可用 `--auto-resume`。

### 后台训练（容器内无 tmux 时）

```bash
cd /workspace/segman/segmentation
conda activate segman

nohup python tools/train.py local_configs/segman_trans/segman_b_trans10k.py \
  --work-dir outputs/trans10k_segman_b \
  --resume-from outputs/trans10k_segman_b/iter_16000.pth \
  --no-validate \
  --cfg-options data.workers_per_gpu=2 \
  > outputs/trans10k_segman_b/train_resume.log 2>&1 &

echo $! > outputs/trans10k_segman_b/train.pid
```

查看进度：`tail -f outputs/trans10k_segman_b/train_resume.log`  
确认进程：`ps aux | grep train.py`  
主机 `nvidia-smi` 显存 ~7GB、利用率接近 100% 通常表示在训（进程名可能不显示 `python`）。

### 多卡（WSL/Linux）

```bash
bash tools/dist_train.sh local_configs/segman_trans/segman_b_trans10k.py 2 \
  --work-dir outputs/trans10k_segman_b
```

### 重建 Docker 容器（数据不丢）

权重与数据在挂载目录 `D:\SegMAN-main\SegMAN`（容器内 `/workspace/segman`），删容器不会删 checkpoint。

WSL 启动示例：

```bash
docker rm -f segman_train 2>/dev/null
docker run --gpus all --name segman_train --shm-size=8g --memory=32g -it \
  -v /mnt/d/SegMAN-main/SegMAN:/workspace/segman \
  segman:v1 bash
```

进容器后必做：`python scripts/fix_mmcv_torch21.py`，再 `--resume-from` 续训。

---

## 第 6 步：训练完成后的验收与评估

> 若训练时使用了 `--no-validate`，**必须**在本节对验证集跑 `tools/test.py` 才能得到 mIoU。  
> 以下命令均在 **`segmentation`** 目录、已 `conda activate segman` 的前提下执行。

### 6.1 确认训练已正常结束

```bash
cd /workspace/segman/segmentation

# 日志末尾应为 iter 80000（或你配置的 max_iters）
tail -30 outputs/trans10k_segman_b/train_resume.log
# 或
tail -30 outputs/trans10k_segman_b/*.log

# 后台任务是否还在（应无 train.py 或已结束）
ps aux | grep train.py
```

**验收**：日志中出现 `Iter [80000/80000]`（或达到 `runner.max_iters`），且无 `Killed` / CUDA OOM 报错。

### 6.2 检查产出文件

```bash
ls -lh outputs/trans10k_segman_b/*.pth
```

| 文件 | 含义 |
|------|------|
| `iter_80000.pth` | 最后一轮权重（若 interval=4000，还会有 76000、72000…） |
| `latest.pth` | 符号链接，指向最近一次保存的 iter |
| `best_mIoU_iter_*.pth` | 训练过程中验证 mIoU 最高的一次（仅当未使用 `--no-validate` 时更新） |

Windows 上也可查看：`D:\SegMAN-main\SegMAN\segmentation\outputs\trans10k_segman_b\`。

### 6.3 验证集评估 mIoU（必做）

**推荐**：优先测训练过程中保存的 best；若用了 `--no-validate`，用最终 iter 权重。

```bash
cd /workspace/segman/segmentation
conda activate segman

# 方式 A：验证集 mIoU（与训练时 evaluation 一致）
python tools/test.py local_configs/segman_trans/segman_b_trans10k.py \
  --checkpoint outputs/trans10k_segman_b/best_mIoU_iter_8000.pth \
  --eval mIoU

# 方式 B：最终 iter 权重（--no-validate 训完后用这个）
python tools/test.py local_configs/segman_trans/segman_b_trans10k.py \
  --checkpoint outputs/trans10k_segman_b/iter_80000.pth \
  --eval mIoU

# 方式 C：latest 指向谁测谁
python tools/test.py local_configs/segman_trans/segman_b_trans10k.py \
  --checkpoint outputs/trans10k_segman_b/latest.pth \
  --eval mIoU
```

终端会打印 **aAcc、mIoU、mAcc** 及各类 **IoU**。建议对 **best** 与 **iter_80000** 各测一次，取 mIoU 更高者作为后续实验权重。

将指标写入工作目录（可选）：

```bash
python tools/test.py local_configs/segman_trans/segman_b_trans10k.py \
  --checkpoint outputs/trans10k_segman_b/iter_80000.pth \
  --eval mIoU \
  --work-dir outputs/trans10k_segman_b/eval_iter_80000
```

### 6.4 查看训练过程日志（若训练中做过验证）

```bash
# 文本日志中的 mIoU 行
grep "Best mIoU" outputs/trans10k_segman_b/*.log

# JSON 日志（可用 analyze_logs 汇总）
python tools/analyze_logs.py outputs/trans10k_segman_b/*.log.json \
  --keys mIoU aAcc --legend segman_b
```

### 6.5 保存可视化结果（可选）

在验证集上导出彩色 mask，便于目视检查透明区域：

```bash
python tools/test.py local_configs/segman_trans/segman_b_trans10k.py \
  --checkpoint outputs/trans10k_segman_b/iter_80000.pth \
  --eval mIoU \
  --show-dir outputs/trans10k_segman_b/vis_val
```

结果在 `outputs/trans10k_segman_b/vis_val/`（与原图同名的着色分割图）。

### 6.6 单张图片推理（可选）

在 Python 中调用 MMSeg API（路径按实际修改）：

```python
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot

config = 'local_configs/segman_trans/segman_b_trans10k.py'
checkpoint = 'outputs/trans10k_segman_b/iter_80000.pth'
img = 'data/trans10k/img_dir/val/val_000000.jpg'

model = init_segmentor(config, checkpoint, device='cuda:0')
result = inference_segmentor(model, img)
show_result_pyplot(model, img, result, opacity=0.5)

# 供 Grounded-SAM / TransFine：取透明前景二值 mask（12 类时 label>0）
# seg = result[0]  # HxW, int
# binary_mask = (seg > 0).astype(np.uint8)
# roi = crop_by_bbox(img, binary_mask)  # 按 mask 外接矩形裁剪
```

### 6.7 发布用精简权重（可选）

去掉优化器状态、仅保留模型权重，便于拷贝与部署：

```bash
python tools/publish_model.py \
  outputs/trans10k_segman_b/iter_80000.pth \
  outputs/trans10k_segman_b/segman_b_trans10k_publish.pth
```

### 6.8 结果记录建议

在项目笔记或论文表格中记录至少：

| 项目 | 示例 |
|------|------|
| 配置 | `segman_b_trans10k.py` |
| 权重 | `iter_80000.pth` / `best_mIoU_iter_8000.pth` |
| val mIoU | 测试命令输出 |
| 训练 setting | 80k iter, bs=2, lr=6e-5, Trans10K 12 类 |

### 6.9 下一步（课题后续）

1. **分割验收**：mIoU、目视 mask 边界；导出 mask 供 ROI 裁剪（系统真正用的是 mask，不是 12 类 logits）。  
2. **对比**：与 debug 2000 iter（`outputs/trans10k_debug/`）的 mIoU 对比，确认长训收益。  
3. **改进分割**：按设计书实现 **LASS** + **MMSCopE**，见《项目实施步骤指南.md》路线 B（步骤 B5～B7）。  
4. **下游（路线 C）**：`SegMAN mask` → **Grounded-SAM** 特征 → **TransFine 细分类** → 抓取仿真；分类步骤见指南 C2～C5，**不由 SegMAN 单独完成**。

正式权重路径供后续配置引用（Docker）：

```text
/workspace/segman/segmentation/outputs/trans10k_segman_b/iter_80000.pth
```

---

## 第 7 步：验证 / 测试（速查）

与第 6.3 相同，最常用的两条：

```bash
cd /workspace/segman/segmentation

python tools/test.py local_configs/segman_trans/segman_b_trans10k.py \
  --checkpoint outputs/trans10k_segman_b/best_mIoU_iter_8000.pth --eval mIoU

python tools/test.py local_configs/segman_trans/segman_b_trans10k.py \
  --checkpoint outputs/trans10k_segman_b/iter_80000.pth --eval mIoU
```

---

## 类别说明（12 类，仅用于训练标注与 mIoU 评测）

下列 ID 来自 Trans10K-v2 多类 mask。**课题端到端系统中，细分类由 Grounded-SAM + TransFine 完成**；SegMAN 训练时使用这些类是为了利用数据集标注并评估分割质量。若只需「是否透明」，可将预测合并为前景，或改用二分类配置。

| ID | 类别 |
|----|------|
| 0 | background |
| 1 | box |
| 2 | bottle |
| 3 | window |
| 4 | eyeglass |
| 5 | freezer |
| 6 | jar/kettle |
| 7 | door |
| 8 | cup |
| 9 | wall |
| 10 | bowl |
| 11 | shelf |

**合并为透明前景（部署示例）**：推理得到 12 类 label map 后，将 `label > 0` 的像素视为透明物体区域，再 crop ROI 交给 Grounded-SAM。

---

## 新增文件一览

| 文件 | 作用 |
|------|------|
| `segmentation/tools/convert_datasets/trans10k.py` | HF → MMSeg 转换 |
| `segmentation/local_configs/_base_/datasets/trans10k.py` | 数据 pipeline |
| `segmentation/local_configs/segman_trans/segman_b_trans10k.py` | SegMAN-B 训练配置 |
| `segmentation/local_configs/segman_trans/segman_b_trans10k_binary.py` | 二分类配置（可选） |

---

## 常见错误

| 报错 | 处理 |
|------|------|
| `No train/validation parquet` | `git lfs pull` |
| `pip install datasets` | 安装 datasets、pyarrow |
| 预训练找不到 | 改 config 里 `pretrained` 绝对路径 |
| CUDA OOM | 减小 crop 或换 `SegMANEncoder_t`（勿用 `samples_per_gpu=1`，BN 会报错） |
| DataLoader Bus error / shm | 见下文「Docker 共享内存」 |

### Docker 共享内存（Bus error）

报错：`insufficient shared memory (shm)`、`DataLoader worker ... Bus error`。

**办法 1（最快）**：减少 worker 或关掉多进程加载

```bash
python tools/train.py local_configs/segman_trans/segman_b_trans10k_debug.py \
  --work-dir outputs/trans10k_debug \
  --cfg-options data.workers_per_gpu=0
```

`workers_per_gpu=2` 一般也可；debug 配置默认已为 2。

**办法 2（推荐长期）**：启动容器时增大 shm

```bash
docker run --gpus all --shm-size=8g -it ...你的镜像...
```

WSL/Docker Desktop：在 Docker 设置里提高资源，或 compose 里写 `shm_size: '8gb'`。

**关于 `fatal: not a git repository`**：仅 mmcv 记录版本信息失败，可忽略，不影响训练。

### 新容器必做：mmcv PyTorch2.1 补丁

```bash
conda activate segman
cd /workspace/segman
python scripts/fix_mmcv_torch21.py
python -c "from mmcv.parallel import collate; print('mmcv ok')"
```

若 `import mmcv` 仍报 `IndentationError`，说明补丁未写入；务必先跑脚本（脚本**不会** import mmcv）。
| `expected np.ndarray (got numpy.ndarray)` | NumPy 2.x 问题：见下文或 `pip install "numpy<2"` |

### NumPy 2.x 与训练报错

若出现 `TypeError: expected np.ndarray (got numpy.ndarray)`，任选其一：

```bash
pip install "numpy<2.0"
```

或已在本仓库修补 `mmseg/datasets/pipelines/formatting.py` 中的 `to_tensor`（需同步到 Docker）。

下一步（你的课题）：① 分割侧实现 LASS / MMSCopE；② 路线 C 用 SegMAN **mask** 衔接 Grounded-SAM + TransFine 做细分类。见《透明物体分割_SegMAN优化设计说明书.md》《项目实施步骤指南.md》。
