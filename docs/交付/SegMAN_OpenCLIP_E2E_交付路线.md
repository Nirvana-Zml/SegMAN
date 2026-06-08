# TransGrasp 双轨交付说明（SegMAN 语义 + M2F 抓取 / OpenCLIP 细分类）

| 项 | 内容 |
| --- | --- |
| 文档版本 | **v2.0 双轨交付** |
| 更新日期 | 2026-06-08 |
| 软件版本 | **V1.0.0** |
| 关联 | `docs/OpenCLIP细分类/OpenCLIP_细分类_完整优化历程与交付说明.md`、`outputs/e2e_improve/f1_execution_summary.md` |
| 文档索引 | `docs/README.md` |
| Manifest | `outputs/e2e_improve/deliver_dual_track_manifest.json` |

---

## 1. 双轨交付总览

交付 **一套软件、两种运行模式**，共享 OpenCLIP 细分类与拒识策略：

| 模式 | ID | 实例来源 | 典型场景 | val match_rate |
| --- | --- | --- | --- | --- |
| **模式 A · 语义** | `semantic` | SegMAN 语义图 → 连通域 | 全图语义理解、标注辅助、离线 SegMAN-ROI 评测 | **59.16%** |
| **模式 B · 抓取** | `grasp` | Mask2Former `iter_40000` | **推荐**：多实例抓取 / E2E、少漏检 | **75.46%** |

```text
                         输入 RGB
                            │
           ┌────────────────┴────────────────┐
           ▼                                 ▼
    【模式 A semantic】                 【模式 B grasp】
    SegMAN iter_6000                    M2F iter_40000
    语义图 → CC                          实例 mask
           │                                 │
           └────────────────┬────────────────┘
                            ▼
              OpenCLIP deliver_classifier_best.pth
              reject_thresholds_p3.json
                            ▼
              pred_class / confidence / grasp|reject
```

**推荐：** 对外交付 **双轨**；生产抓取默认 **模式 B**，语义可视化与课题验收保留 **模式 A**。

---

## 2. 正式交付件清单

### 2.1 共享模块（两模式共用）

| 模块 | 路径 |
| --- | --- |
| **细分类** | `outputs/openclip_classifier/deliver_classifier_best.pth` |
| 分类源 | `outputs/openclip_classifier/p3_p1_hardmining/best.pth` |
| 分类 manifest | `outputs/openclip_classifier/deliver_p3/deliver_manifest.json` |
| **拒识** | `transgrasp/classification/configs/reject_thresholds_p3.json` |
| **E2E 核心** | `transgrasp/pipelines/segment_and_classify.py` |
| **交付入口（封装）** | `app/run_semantic_e2e.py`、`app/run_grasp_e2e.py` |
| **验收脚本** | `scripts/run_deliver_semantic_e2e.sh`、`scripts/run_deliver_grasp_e2e.sh` |

### 2.2 模式 A：SegMAN 语义分割

| 项 | 路径 |
| --- | --- |
| Checkpoint | `segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth` |
| Config | `segmentation/local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py` |
| val mIoU | **81.80%** |
| 环境 | **`segman`** |

### 2.3 模式 B：Mask2Former 实例（抓取轨）

| 项 | 路径 |
| --- | --- |
| Checkpoint | `segmentation/outputs/m2f_trans10k_pseudo/iter_40000.pth` |
| Config | `segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py` |
| COCO segm mAP | **62.5%**（40k） |
| 环境 | **`segman_mmdet`**（推理）；summarize 可用 `segman` |

### 2.4 不纳入交付

| 项 | 原因 |
| --- | --- |
| `m2f_roi_adapt_v1/best.pth` | P1 实测 E2E cls 下降 |
| `deliver_classifier_t2_archived.pth` | 仅回滚 |
| P4 等实验 ckpt | 未 promote |

---

## 3. 部署命令

### 3.1 模式 A — 语义 + 细分类（推荐封装入口）

```bash
conda activate segman
cd /path/to/SegMAN

python app/run_semantic_e2e.py \
  --image path/to/image.jpg \
  --out-dir outputs/deliver_run/semantic_demo
```

等价底层命令：

```bash
python transgrasp/pipelines/segment_and_classify.py \
  --instance-source semantic \
  --seg-checkpoint segmentation/outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth \
  --seg-config segmentation/local_configs/segman_trans/segman_b_trans10k_lass_balanced_v2.py \
  --cls-checkpoint outputs/openclip_classifier/deliver_classifier_best.pth \
  --class-thresholds transgrasp/classification/configs/reject_thresholds_p3.json \
  --min-area 128 --nms-iou 0.5 --bbox-pad 0.15 \
  --image path/to/image.jpg --out-dir outputs/deliver_run/semantic_demo
```

### 3.2 模式 B — M2F 实例 + 细分类（抓取推荐）

```bash
conda activate segman_mmdet
cd /path/to/SegMAN

python app/run_grasp_e2e.py \
  --image path/to/image.jpg \
  --out-dir outputs/deliver_run/grasp_demo
```

等价底层命令：

```bash
python transgrasp/pipelines/segment_and_classify.py \
  --instance-source m2f \
  --m2f-checkpoint segmentation/outputs/m2f_trans10k_pseudo/iter_40000.pth \
  --m2f-config segmentation/local_configs/mask2former/m2f_trans10k_pseudo_instances.py \
  --m2f-score-thresh 0.30 \
  --cls-checkpoint outputs/openclip_classifier/deliver_classifier_best.pth \
  --class-thresholds transgrasp/classification/configs/reject_thresholds_p3.json \
  --min-area 128 --nms-iou 0.5 --bbox-pad 0.15 \
  --image path/to/image.jpg --out-dir outputs/deliver_run/grasp_demo
```

### 3.3 val 全量验收（双轨）

```bash
# 模式 A
bash scripts/run_deliver_semantic_e2e.sh

# 模式 B（需 segman_mmdet）
bash scripts/run_deliver_grasp_e2e.sh
```

### 3.4 环境安装

| 环境 | 用途 | 脚本 |
| --- | --- | --- |
| `segman` | 模式 A + OpenCLIP + summarize | 项目 Conda 默认环境 |
| `segman_mmdet` | 模式 B M2F 推理 | `scripts/setup_f1_mmdet_env.sh` |

---

## 4. 交付指标对比（Trans10K val，相同后处理）

| 指标 | 模式 A `f1_b1_ref` | 模式 B `f1_m2f_e2e` |
| --- | --- | --- |
| **match_rate** | 59.16% | **75.46%** |
| pred_gt_ratio | 1.0457 | 1.0451 |
| **e2e_top1_on_matched** | **84.59%** | 81.78% |
| strict_e2e_all_gt | 50.05% | **61.71%** |
| wall_match | 47.91% | **68.53%** |
| grasp_rate_on_matched | 79.86% | ~80% |

**解读：**

- **模式 B** 显著减少实例漏检（+16pp match），适合抓取与严格端到端。
- **模式 A** 匹配后分类略高，适合「语义分割 + 细分类」课题表述与 SegMAN-ROI 离线指标。
- 分类器为 **同一 deliver**；勿使用 `m2f_roi_adapt_v1`。

### 4.1 模式 A 后处理优化记录（Phase 1，2026-06-08）

2026-06-08 完成语义轨 **后处理参数 sweep**（不改 `iter_6000.pth`），结论：

| 项 | 值 |
| --- | --- |
| 正式 deploy | **维持 B1**，match **59.16%**（本节 §4 表格不变） |
| Phase 1 最优（未上线） | **60.84%**（TTA + per-class IoU 0.22，+1.68pp） |
| 主目标 63% | ❌ 未达成 |
| 瓶颈 | 实例漏检（miss ~90%），CC/merge 无法根治 |

详报：`docs/优化SegMANmatch_rate/Phase1_后处理优化实验报告.md`  
台账：`outputs/match_improve/phase1_summary.json`

---

## 5. 交付包目录建议（zip 结构）

```text
TransGrasp-V1.0.0/
├── README_交付说明.md              # 本文档副本
├── app/
│   ├── run_semantic_e2e.py
│   └── run_grasp_e2e.py
├── models/                         # 或 MODELS_README + 下载链接
├── transgrasp/
├── segmentation/                   # 必要配置与工具
├── scripts/
│   ├── run_deliver_semantic_e2e.sh
│   ├── run_deliver_grasp_e2e.sh
│   └── setup_f1_mmdet_env.sh
├── docs/
│   └── SegMAN_OpenCLIP_E2E_交付路线.md
└── outputs/e2e_improve/
    └── deliver_dual_track_manifest.json
```

---

## 6. 训练数据（配套）

| 数据集 | 路径 | 用途 |
| --- | --- | --- |
| GT ROI | `data/trans10k_roi_gt/` | 分类训练 |
| SegMAN ROI | `data/trans10k_roi_segman/` | 模式 A 离线评测 |
| M2F ROI（P1 实验） | `data/trans10k_roi_m2f/` | 非 deliver 分类 |

---

## 7. 交付自检清单

- [ ] 模式 A：`iter_6000.pth` + `segman` 环境，`run_semantic_e2e.py` 单图可跑通  
- [ ] 模式 B：`iter_40000.pth` + `segman_mmdet`，`run_grasp_e2e.py` 单图可跑通  
- [ ] `deliver_classifier_best.pth`、`reject_thresholds_p3.json` 存在  
- [ ] val 评测：A 与 `f1_b1_ref`、B 与 `f1_m2f_e2e` 指标同量级  
- [ ] `deliver_dual_track_manifest.json` 版本号为 V1.0.0  

刷新 manifest：`bash scripts/promote_dual_track_deliver.sh`

---

## 8. 软著与验收说明（摘要）

- 软件著作权建议按 **1 套软件** 登记，说明书分「模式 A / 模式 B」两章。  
- 详见 `docs/交付/双轨交付与软著材料清单.md`（说明书大纲、源代码页选列表）。
