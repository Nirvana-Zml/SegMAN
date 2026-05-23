# 透明物体 SegMAN 优化与分类抓取一体化系统设计说明书

| 项目 | 内容 |
|------|------|
| 文档版本 | v2.0 |
| 编写日期 | 2026-05-16 |
| 基线模型 | SegMAN（CVPR 2025，MMSegmentation v0.30.0） |
| 数据集 | Trans10K-v2（透明物体分割与细分类标注） |
| 系统目标 | 高质量透明物体语义分割 → 细分类 → 用户交互式定位/分类/抓取一体化 |
| 核心模块 | 编码器 **LASS**、解码器 **MMSCopE**、细分类子模型 **TransFine**、抓取仿真 **ASGrasp + PyBullet** |

---

## 0. 系统总览

本系统按 **三阶段流水线** 构建，对应总体技术路线如下：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 阶段一：数据准备与预处理                                                        │
│  获取 Trans10K-v2 → 数据增强（反射模拟/光照变化/边界模糊）→ 构建训练数据集          │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 阶段二：SegMAN 网络优化                                                        │
│  编码器 LASS（低纹理注意力 + 反射抑制）→ 解码器 MMSCopE（边界掩码 + 多尺度边界卷积） │
│  输出：高质量透明物体语义分割 mask / 实例区域                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 阶段三：透明物体细分类与抓取一体化系统                                            │
│  Grounded-SAM 等大模型特征提取 → TransFine 细分类训练/推理                       │
│  用户交互指定目标 → ASGrasp 六自由度抓取姿态 → PyBullet 仿真 → 参数迭代优化        │
└──────────────────────────────────────────────────────────────────────────────┘
```

**阶段二** 与 **阶段三** 通过 **分割 mask + 裁剪 ROI** 衔接：分割结果提供空间定位，细分类与抓取模块在 mask 约束区域内工作，避免背景干扰。

---

## 1. 背景与动机

### 1.1 应用背景

透明物体在工业分拣、服务机器人、实验室自动化等场景中广泛存在。单一语义分割无法区分“玻璃瓶 / 塑料杯 / 玻璃碗”等细粒度类别，也无法直接驱动机械臂完成抓取。本设计在 SegMAN 分割能力之上，扩展 **细分类** 与 **仿真抓取** 闭环，形成可交互的端到端系统。

### 1.2 透明物体视觉与任务难点

| 现象 | 对分割的影响 | 对分类/抓取的影响 |
|------|----------------|-------------------|
| 低纹理、高透光 | 特征弱、易漏检 | 类间外观相似，需高维判别特征 |
| 镜面/折射反射 | 特征与背景混淆 | 深度/点云缺失时抓取点不稳定 |
| 边界半透明、混合像素 | 轮廓粗糙 | 抓取接触区域估计偏差大 |
| 类别多样 | — | 需细粒度标签与专用分类头 |

SegMAN 通过 **Neighborhood Attention + VSSM/SS2D** 建模多尺度上下文，在通用分割上表现优异；本方案针对透明场景增加 **LASS / MMSCopE**，并引入 **大模型特征 + 轻量细分类子网络** 及 **ASGrasp 仿真迭代**。

### 1.3 设计原则

1. **模块化**：数据、分割、分类、抓取四层解耦，接口以 mask / 类别 ID / 6D pose 传递。
2. **最小侵入**：SegMAN 主干保留预训练加载能力，LASS/MMSCopE 增量扩展。
3. **可交互**：支持用户点选/框选/类别名指定目标，再触发分类确认与抓取规划。
4. **仿真先行**：PyBullet 中迭代优化抓取参数，再映射到真机（可选扩展）。

### 1.4 术语对照

| 设计术语 | 代码/实现 | 说明 |
|----------|-----------|------|
| SS2D | `VSSM` | Cross-Scan + Selective Scan |
| LASS | `SegMANEncoderLASS` | 低纹理注意力 + 反射抑制编码器 |
| MMSCopE | `SegMANDecoderMMSCopE` | 边界增强解码器 |
| TransFine | 细分类子模型（待实现） | 透明物体类型判别 |
| ASGrasp | 六自由度抓取规划模块 | 抓取姿态估计与仿真评估 |
| 一体化系统 | `TransGraspUI`（建议包名） | 交互 + 分割 + 分类 + 仿真 |

---

## 2. 阶段一：数据准备与预处理

### 2.1 数据集：Trans10K-v2

**Trans10K-v2** 为本项目主数据集，提供：

| 内容 | 用途 |
|------|------|
| RGB 图像 | 分割与分类输入 |
| 透明物体语义/实例 mask | 训练 LASS、MMSCopE；生成边界 GT、背景 mask |
| 细粒度类别标注 | 构建细分类训练集（见 §6） |
| train/val 划分 | 分割与分类分别评估 |

**目录结构建议**

```
data/Trans10K-v2/
├── train/
│   ├── image/
│   ├── mask/              # 语义或实例 mask
│   └── meta.csv           # image_id, class_name, class_id
├── val/
│   └── ...
└── class_names.json       # 类别表：如 bottle, cup, bowl, ...
```

### 2.2 数据增强策略

在 Trans10K-v2 基础上构建 **分割训练集** 与 **分类训练集**（可同源、不同 pipeline）：

| 增强类型 | 方法 | 针对问题 |
|----------|------|----------|
| **反射模拟** | 随机粘贴背景 patch 至物体 ROI、Alpha 混合 | 训练 RSM 反射抑制 |
| **光照变化** | 亮度/对比度/色温 jitter、随机阴影 | 提升光照鲁棒性 |
| **边界模糊** | 对 mask 做高斯模糊再重二值化、形态学腐蚀膨胀 | 训练 BPM/MSBEC 边界分支 |
| 几何增强 | 翻转、缩放、裁剪（保持 mask 对齐） | 通用泛化 |
| 分类专用 | 同类不同实例裁剪、CutMix（可选） | 细分类泛化 |

**构建流程**

```
Trans10K-v2 原始样本
    → 增强 pipeline（反射/光照/边界）
    → seg_dataset/（MMSeg 格式：img + ann）
    → cls_dataset/（ROI 裁剪 + 类别标签，见 §6.2）
```

### 2.3 与 MMSegmentation 的对接

- 分割：`segmentation/tools/convert_datasets/` 下新增 `trans10k.py`，生成 MMSeg 索引文件。
- 配置：`segmentation/local_configs/segman_trans/segman_b_trans10k.py`，`num_classes` 与 Trans10K 透明类定义一致。

---

## 3. 阶段二：SegMAN 网络优化（高质量语义分割）

阶段二输出 **高质量语义分割结果**（像素级 mask + 可选实例 ID），作为阶段三的 **空间先验** 与 **ROI 裁剪依据**。

### 3.0 阶段内数据流

```
输入图像 I
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  SegMANEncoder-LASS（编码器）                              │
│  stem → Stage1~3 [NA + SS2D + LTAB + RSM]                │
│       → Stage4 [Global Attn] → {c1, c2, c3, c4}         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  SegMANDecoder-MMSCopE（解码器）                           │
│  MLP 融合 → BPM + MSBEC → forward_winssm → cls_seg       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
分割 mask M_seg、边界图 P_bd（可选中间量）
```

---

## 4. 编码器改进：LASS

**LASS**（**L**ow-texture **A**ware **S**SM with reflection **S**uppression）：在 Stage 1～3 的 SS2D（`VSSM`）路径上叠加 **低纹理区域注意力（LTAB）** 与 **反射抑制（RSM）**。

### 4.1 插入位置

改造 `Attention.forward` 非 `global_mode` 分支：

```
v_r = NA 输出
v   = dwconv(v_r) → SiLU → [RSM] → VSSM/SS2D → [LTAB 加权] → norm
x   = v + v_r
```

仅 Stage 1～3 启用；Stage 4 保持全局注意力。

### 4.2 子模块 A：低纹理区域注意力（LTAB）

**目标**：对 SS2D 输出施加空间权重，强化 **无纹理但属于透明物体** 的区域。

**纹理图**（在 `F_ssm` 上计算）：

```
G_x = |∂F/∂x|,  G_y = |∂F/∂y|
T   = AvgPool( ||G_x|| + ||G_y|| )
T_norm = (T - μ) / (σ + ε)
W_lt = σ( α · (τ - T_norm) )
F_out = F_ssm ⊙ (1 + β · W_lt^c)
```

### 4.3 子模块 B：反射抑制（RSM）

**目标**：SS2D 扫描前削弱背景反射分量，残差保留微弱物体特征。

```
F_bg   = M_bg ⊙ v
μ_bg   = MaskedGAP(F_bg)
F_refl = M_bg ⊙ Broadcast(μ_bg)
F_clean = v - γ · F_refl
F_in = F_clean + δ · v    → 送入 VSSM
```

| 掩码来源 | 说明 |
|----------|------|
| 训练 | `M_bg = 1 - Dilate(Y_fg)`，或 BgHead 预测 |
| 推理 | BgHead(F) → `M_bg` |

**流程顺序**：`v → RSM → SS2D → LTAB → norm → +v_r`。

### 4.4 配置示例

```python
backbone=dict(
    type='SegMANEncoderLASS',
    lass_cfg=dict(
        enable_stages=[0, 1, 2],
        ltab=dict(texture='gradient', beta_init=0.1),
        rsm=dict(enable_stages=[1, 2], gamma_init=0.5, delta_init=0.5,
                 bg_mask_mode='gt+pred'),
    ),
)
```

---

## 5. 解码器改进：MMSCopE

**MMSCopE**（**M**ulti-scale **M**ask-guided **S**emantic fusion with **Co**ntour-aware **P**ixel r**E**finement）：引入 **边界注意力掩码** 与 **多尺度边界增强卷积（MSBEC）**。

### 5.1 边界概率图（BPM）

```
BPM: Conv3×3 → BN → ReLU → Conv3×3 → Conv1×1 → Sigmoid  →  P_bd
Y_bd = Dilate(Y) - Erode(Y)  或  |∇Y|
L_bd = BCE / Dice(P_bd, Y_bd)
W_bd = σ( Conv1×1(Concat(P_bd, Pool(P_bd))) )
_c2' = _c2 ⊙ (1 + η · W_bd)
```

### 5.2 多尺度边界增强（MSBEC）

| 分支 | 分辨率 | 操作 |
|------|--------|------|
| S0 | H×W | DWConv3×3 + Conv1×1 |
| S1 | H/2 | stride=2 + DWConv |
| S2 | H/4 | stride=4 + DWConv |

```
F_ref = Conv( Concat(Upsample(S0,S1,S2), P_bd) )
F_fuse = Conv(Concat(F_sem, F_ref)) + F_sem
```

可选边界修正：`logits = cls_seg(F_fuse) + λ · (Conv3×3(F_fuse) ⊙ P_bd)`。

### 5.3 输出与下游接口

| 输出 | 形状 | 下游用途 |
|------|------|----------|
| `M_seg` | B×H×W | 细分类 ROI 裁剪、抓取物体点云掩膜 |
| `P_bd` | B×1×H×W | 边界加权、抓取接触区细化 |
| 多尺度特征（可选导出） | — | 与大模型特征融合（可选扩展） |

---

## 6. 阶段三（上）：透明物体细分类

### 6.1 设计目标

在分割得到的 **透明物体区域** 内，区分 Trans10K-v2 定义的多种细粒度类别（如不同材质/形状的瓶子、杯子等），为抓取策略选择与用户确认提供 **类别语义**。

### 6.2 基于大模型的高维判别特征提取

采用 **Grounded-SAM**（或 SAM + Grounding DINO 组合）作为 **冻结特征提取器**，在分割 mask 约束下提取判别性表征：

```
输入：原图 I + 分割 mask M_seg（+ 可选用户文本提示）
    │
    ▼
ROI 裁剪：I_roi = Crop(I, bbox(M_seg))，可选 mask 抠图
    │
    ▼
Grounded-SAM / SAM Image Encoder
    →  F_fm ∈ R^{D_fm}（如 256/1024 维全局或 patch 聚合特征）
    │
    ▼
与 SegMAN 特征融合（可选）：
    F_roi_seg = GAP(Encoder_LASS(I) ⊙ M_seg)
    F_cat = Concat(F_fm, F_roi_seg)
```

| 特征源 | 维度（示例） | 优势 |
|--------|--------------|------|
| Grounded-SAM 图像编码 | 256～1024 | 开放词汇、语义判别强 |
| SegMAN-LASS ROI 特征 | 256～560 | 针对透明区域优化、与分割一致 |
| 几何描述子（可选） | 16～32 | 宽高比、面积、轮廓矩 |

**推荐默认**：`F = MLP(Concat(F_fm, F_roi_seg))`，推理时 Grounded-SAM **冻结**，仅训练轻量头。

### 6.3 细分类训练数据集构建

基于 Trans10K-v2 **类别标注** 自动构建：

```
对每张图 (I, Y, class_id)：
    1. 用 GT mask 或 M_seg 得到连通域
    2. 按实例裁剪 ROI，padding 10%～20%
    3. 记录 (roi_path, class_id, class_name, split)
    4. 提取 F_fm（离线缓存 .npy 可加速训练）
```

**cls_dataset 结构**

```
data/Trans10K-v2-cls/
├── train/
│   ├── roi/
│   └── features/          # 可选预计算 F_fm
├── val/
└── labels.csv             # roi_id, class_id
```

**类别不平衡**：加权 CE、类均衡采样、focal loss。

### 6.4 细分类子模型：TransFine

**TransFine**：轻量 MLP / 小 Transformer 分类头，输入 `F_cat`，输出 `K` 类 logits。

```
TransFine:
  Input:  F_cat ∈ R^D
  Hidden: Linear(D, 512) → BN → ReLU → Dropout
          Linear(512, 256) → BN → ReLU
  Output: Linear(256, K) → softmax
```

| 项目 | 建议 |
|------|------|
| 损失 | CrossEntropy + label smoothing(0.1) |
| 优化器 | AdamW, lr=1e-3, 50～100 epoch |
| 指标 | Top-1 Acc、每类 Recall、混淆矩阵 |
| 与分割联合 | 先固定分割权重，仅训 TransFine；再可选联合微调 ROI 分支 |

**推理输出**：`(class_id, class_name, confidence)`，与用户交互模块联动。

### 6.5 模块接口

```python
class FeatureExtractor:
    """Grounded-SAM + 可选 SegMAN ROI 特征。"""
    def extract(self, image, mask) -> Tensor: ...

class TransFineClassifier(nn.Module):
    def forward(self, feat: Tensor) -> Tensor: ...  # logits

class TransFinePipeline:
    def predict(self, image, seg_mask) -> dict:
        # return class_id, class_name, score, roi_bbox
```

---

## 7. 阶段三（下）：分类与抓取一体化系统

### 7.1 系统功能概述

构建 **用户交互驱动的透明物体分类与抓取一体化系统**，实现：

1. **精准定位**：SegMAN-LASS + MMSCopE 分割 + 用户点选/框选；
2. **精准分类**：TransFine 对指定 ROI 细分类，结果可人工确认；
3. **精准抓取**：ASGrasp 估计 6-DOF 抓取位姿，PyBullet 仿真验证并迭代优化参数。

### 7.2 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     用户交互层 (TransGraspUI)                      │
│  图像上传 / 摄像头 │ 点击选实例 │ 输入类别名 │ 确认抓取 │ 显示结果    │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────────┐   ┌──────────────────┐
│ SegMAN 分割  │    │ TransFine 分类   │   │ ASGrasp 抓取规划  │
│ LASS+MMSCopE│    │ Grounded-SAM特征 │   │ 6-DOF pose       │
└──────┬──────┘    └────────┬────────┘   └────────┬─────────┘
       │                    │                      │
       └────────────────────┴──────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ PyBullet 仿真    │
                    │ 抓取成功判定      │
                    └────────┬────────┘
                             │ 失败/次优
                             ▼
                    抓取参数迭代优化 ──► 回写 ASGrasp
```

### 7.3 用户交互流程

| 步骤 | 用户操作 | 系统行为 |
|------|----------|----------|
| 1 | 上传/采集场景图像 | 运行分割，叠加透明 mask 可视化 |
| 2 | 点击目标或框选区域 | 关联最近实例 mask；若无分割则提示 |
| 3 | （可选）输入类别关键词 | Grounded-SAM 文本引导辅助定位 |
| 4 | 查看分类结果 | TransFine 输出类别与置信度；低置信度提示确认 |
| 5 | 确认抓取 | 触发 ASGrasp + PyBullet |
| 6 | 查看仿真结果 | 成功/失败、评分、优化后位姿；可导出到真机 |

### 7.4 ASGrasp 六自由度抓取

**输入**

| 输入项 | 来源 |
|--------|------|
| RGB / 深度（可选） | 相机或仿真渲染 |
| 物体 mask `M_seg` | 分割模块 |
| 类别 `class_id` | TransFine（用于先验抓取模板或参数组） |
| 物体位姿初值 | mask 质心 + 主方向 |

**输出**

- 夹爪 6-DOF 位姿：`T_grasp = (R, t)` 或 `(x,y,z, roll,pitch,yaw)`
- 抓取宽度、接近方向、预抓取/后撤距离

**与分类联动**：不同透明类别可配置 **抓取参数表**（夹爪开口、approach 角度、力阈值）。

### 7.5 PyBullet 仿真平台

**场景搭建**

```
PyBullet 环境
├── 平面/桌面 URDF
├── 机械臂 URDF（如 Panda、UR5）
├── 平行夹爪
├── 透明物体 mesh（按 class_id 加载不同 CAD/简模）
└── 可选：RGB-D 虚拟相机
```

**仿真测试流程**

```
1. 根据 M_seg 与 T_grasp 放置物体与机械臂初态
2. 执行 approach → close gripper → lift
3. 判定指标：
   - 是否接触物体（collision）
   - _lift 后物体位移 / 是否滑落
   - 夹持稳定性（可选力矩阈值）
4. 评分 S_grasp ∈ [0, 1]
```

### 7.6 抓取参数迭代优化（闭环）

当仿真评分低于阈值时，对抓取超参迭代搜索并回写：

```
θ = [approach_dist, gripper_width, grasp_depth, angle_offset, ...]
repeat:
    T' = ASGrasp(I, M_seg, class_id, θ)
    S  = PyBulletEvaluate(T')
    θ  = Update(θ, S)    # 网格搜索 / 贝叶斯优化 / CEM
until S > S_thresh or max_iters
```

| 参数 | 典型范围 | 说明 |
|------|----------|------|
| `approach_dist` | 0.02～0.15 m | 预抓取接近距离 |
| `gripper_width` | 按物体 bbox 缩放 | 夹爪开口 |
| `angle_offset` | ±15° | 绕接近轴旋转 |
| `grasp_depth` | ROI 法向深度偏移 | 透明物体易滑移，需微调 |

**输出**：最优 `θ*`、`T_grasp*`、仿真视频/截图，供真机复现。

### 7.7 一体化系统目录规划

```
transgrasp/                          # 阶段三工程根目录（建议新建）
├── ui/
│   └── app.py                       # Gradio / PyQt 交互界面
├── segmentation/
│   └── infer_segman.py              # 调用 MMSeg 分割
├── classification/
│   ├── extract_features.py          # Grounded-SAM 特征
│   ├── transfine.py                 # 细分类模型
│   └── train_transfine.py
├── grasping/
│   ├── asgrasp_wrapper.py           # ASGrasp 接口封装
│   ├── pybullet_env.py              # 仿真环境
│   └── optimize_grasp.py            # 参数迭代
├── configs/
│   ├── trans10k.yaml
│   └── grasp_class_prior.yaml       # 每类抓取先验
└── pipelines/
    └── run_interactive.py           # 端到端入口
```

---

## 8. 损失函数与训练策略（分割 + 分类）

### 8.1 分割损失

```
L_seg_total = L_seg + λ_bd · L_bd + λ_bg · L_bg
```

| 项 | 权重建议 |
|----|----------|
| `L_seg` | 1.0 |
| `L_bd` | 0.2～0.5 |
| `L_bg` | 0.1～0.3 |

### 8.2 分割训练阶段

| 阶段 | 内容 |
|------|------|
| S0 | 加载 SegMAN 预训练；冻结 Stage 4 |
| S1 | 训练 LASS + BPM |
| S2 | 解冻 MMSCopE，端到端微调 |
| S3 | Trans10K-v2 全量精调；`bg_mask_mode`: gt → pred |

### 8.3 细分类训练阶段

| 阶段 | 内容 |
|------|------|
| C0 | 离线提取 Grounded-SAM 特征缓存 |
| C1 | 仅训练 TransFine（冻结 SAM） |
| C2 | （可选）SegMAN ROI 分支小 lr 联合微调 |

### 8.4 抓取模块

- ASGrasp / 仿真 **无需梯度训练**（本阶段以参数搜索为主）；若有抓取网络权重可加载预训练。
- 迭代优化在 **验证集场景子集** 上标定默认 `θ`  per class。

---

## 9. 实现规划与里程碑

### 9.1 代码仓库分工

| 路径 | 内容 |
|------|------|
| `segmentation/mmseg/models/backbones/segman_encoder_lass.py` | LASS 编码器 |
| `segmentation/mmseg/models/decode_heads/segman_decoder_mmscope.py` | MMSCopE 解码器 |
| `segmentation/local_configs/segman_trans/` | Trans10K 分割配置 |
| `segmentation/tools/convert_datasets/trans10k.py` | 数据转换 |
| `transgrasp/` | 分类 + 抓取 + UI（新建） |

### 9.2 里程碑

| ID | 阶段 | 交付物 | 验收 |
|----|------|--------|------|
| M0 | 数据 | Trans10K-v2 转换 + 增强 pipeline | MMSeg 可训练 |
| M1 | 分割 | AttentionLASS | 前向正确、权重可加载 |
| M2 | 分割 | SegMANEncoderLASS + config | 透明类 mIoU ↑ |
| M3 | 分割 | SegMANDecoderMMSCopE | Boundary F-score ↑ |
| M4 | 分类 | cls 数据集 + Grounded-SAM 特征 | 缓存可复用 |
| M5 | 分类 | TransFine 训练 | Top-1 Acc 达预期 |
| M6 | 系统 | PyBullet 场景 + ASGrasp 封装 | 单次仿真可跑通 |
| M7 | 系统 | 参数迭代 + UI | 用户指定物体完成分类+抓取闭环 |

### 9.3 消融实验

**分割（E0～E5）**

| ID | 编码器 | 解码器 |
|----|--------|--------|
| E0 | SegMAN | SegMAN |
| E3 | LASS | 基线 |
| E5 | LASS | MMSCopE |

**分类（C0～C2）**

| ID | 特征 | 说明 |
|----|------|------|
| C0 | 仅 ResNet ROI | 基线 |
| C1 | 仅 Grounded-SAM | 大模型特征 |
| C2 | SAM + SegMAN ROI | 完整 TransFine |

**抓取（G0～G1）**

| ID | 策略 |
|----|------|
| G0 | 固定 θ，无迭代 |
| G1 | PyBullet + 参数迭代 |

---

## 10. 复杂度、风险与依赖

### 10.1 软件依赖

| 组件 | 用途 |
|------|------|
| MMSegmentation v0.30 | 分割训练/推理 |
| NATTEN、selective_scan CUDA | SegMAN 算子 |
| Grounded-SAM / SAM | 特征提取 |
| PyBullet | 机械臂仿真 |
| ASGrasp | 六自由度抓取（按原论文/官方实现接入） |
| Gradio 或 PyQt | 用户界面 |

### 10.2 风险与缓解

| 风险 | 缓解 |
|------|------|
| 分割失败导致分类/抓取错误 | UI 强制用户确认 mask；低 IoU 警告 |
| Grounded-SAM 推理慢 | 特征离线缓存；ROI 小图推理 |
| 透明物体仿真物理不准 | 简化接触模型；迭代优化补偿 |
| 类间抓取策略差异大 | `grasp_class_prior.yaml`  per-class 先验 |

---

## 11. 配置示例

### 11.1 分割（Trans10K-v2）

```python
# segmentation/local_configs/segman_trans/segman_b_trans10k.py
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SegMANEncoderLASS',
        pretrained='pretrained/SegMAN_Encoder_b.pth.tar',
        lass_cfg=dict(
            enable_stages=[0, 1, 2],
            ltab=dict(texture='gradient', channel_wise=True, beta_init=0.1),
            rsm=dict(enable_stages=[1, 2], gamma_init=0.5, delta_init=0.5,
                     bg_mask_mode='gt+pred'),
        ),
    ),
    decode_head=dict(
        type='SegMANDecoderMMSCopE',
        in_channels=[96, 160, 364, 560],
        in_index=[0, 1, 2, 3],
        channels=180,
        feat_proj_dim=320,
        num_classes=2,  # 按 Trans10K 透明/背景或多类设定
        mmscope_cfg=dict(boundary_loss_weight=0.4, refine_lambda=0.1),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
)
```

### 11.2 细分类

```yaml
# transgrasp/configs/transfine.yaml
num_classes: 10          # 与 Trans10K-v2 细类数一致
feature_dim: 1280        # F_fm + F_roi_seg
sam_model: grounded_sam
seg_checkpoint: outputs/segman_b_trans10k/latest.pth
train:
  epochs: 80
  batch_size: 64
  lr: 1.0e-3
```

### 11.3 抓取先验（示例）

```yaml
# transgrasp/configs/grasp_class_prior.yaml
bottle:
  gripper_width_scale: 0.6
  approach_angle_deg: 30
cup:
  gripper_width_scale: 0.8
  approach_angle_deg: 45
```

---

## 12. 参考文献

1. SegMAN (CVPR 2025).  
2. Trans10K / Trans10K-v2：透明物体分割与检测数据集。  
3. Grounded-SAM：开放词汇分割与特征。  
4. VMamba / SS2D：二维选择性扫描。  
5. ASGrasp：透明/困难物体抓取相关 work（实现时对照原文献接口）。  
6. PyBullet：物理仿真与机械臂控制。

---

## 13. 修订记录

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-05-16 | 初稿：LASS + MMSCopE |
| v2.0 | 2026-05-16 | 扩展：Trans10K-v2 数据管线；Grounded-SAM + TransFine 细分类；ASGrasp + PyBullet 一体化交互系统；三阶段总览 |

---

**附录 A：符号表**

| 符号 | 含义 |
|------|------|
| `M_seg` | 语义分割 mask |
| `P_bd`, `W_bd` | 边界概率图 / 边界注意力权重 |
| `F_fm` | 大模型（Grounded-SAM）特征 |
| `F_roi_seg` | SegMAN ROI 池化特征 |
| `T_grasp` | 6-DOF 抓取位姿 |
| `θ` | 抓取超参向量 |

**附录 B：阶段交付物清单**

| 阶段 | 交付物 |
|------|--------|
| 一 | Trans10K-v2 MMSeg 数据集、增强脚本、cls 数据集 |
| 二 | SegMAN-LASS + MMSCopE 权重、分割推理 API |
| 三 | TransFine 权重、TransGraspUI、PyBullet 场景、抓取优化脚本 |
