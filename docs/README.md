# SegMAN 项目文档索引

文档已按任务类型归类至子目录。路径均相对 `SegMAN/` 根目录。

---

## 目录结构

```text
docs/
├── README.md                          # 本索引
├── 交付/                              # 双轨正式交付、软著
├── OpenCLIP细分类/                    # 细分类训练与优化
├── 路线B_SegMAN分割/                  # LASS/MMSCopE、平衡微调
├── 路线C_细分类与抓取/                # 细分类→抓取路线
├── Trans10K/                          # 数据集训练与评测
├── E2E端到端/                         # E2E、F1、match 优化
├── 优化SegMANmatch_rate/              # 语义轨 match_rate 提升计划
└── 项目总览/                          # 项目实施总指南
```

---

## 交付（双轨 V1.0）

| 文档 | 说明 |
| --- | --- |
| [交付/SegMAN_OpenCLIP_E2E_交付路线.md](交付/SegMAN_OpenCLIP_E2E_交付路线.md) | **双轨正式交付说明** |
| [交付/双轨交付与软著材料清单.md](交付/双轨交付与软著材料清单.md) | 软著 / zip / 验收 |

---

## OpenCLIP 细分类

| 文档 | 说明 |
| --- | --- |
| [OpenCLIP细分类/OpenCLIP_细分类_完整优化历程与交付说明.md](OpenCLIP细分类/OpenCLIP_细分类_完整优化历程与交付说明.md) | P0～P4 + 方案 B 结题 |
| [OpenCLIP细分类/OpenCLIP_细分类_未达80%原因与优化方案.md](OpenCLIP细分类/OpenCLIP_细分类_未达80%原因与优化方案.md) | 未达 80% 根因与方案 |
| [OpenCLIP细分类/OpenCLIP_细分类训练与优化指南.md](OpenCLIP细分类/OpenCLIP_细分类训练与优化指南.md) | 训练复现指南 |

---

## 路线 B — SegMAN 分割

| 文档 | 说明 |
| --- | --- |
| [路线B_SegMAN分割/透明物体分割_SegMAN优化设计说明书.md](路线B_SegMAN分割/透明物体分割_SegMAN优化设计说明书.md) | 系统设计说明书 |
| [路线B_SegMAN分割/路线B_LASS_MMSCopE_实施清单.md](路线B_SegMAN分割/路线B_LASS_MMSCopE_实施清单.md) | LASS/MMSCopE 实施清单 |
| [路线B_SegMAN分割/路线B_平衡微调方案.md](路线B_SegMAN分割/路线B_平衡微调方案.md) | balanced_v2 微调 |
| [路线B_SegMAN分割/路线B_基线_fix5k_balanced10k_对比分析.md](路线B_SegMAN分割/路线B_基线_fix5k_balanced10k_对比分析.md) | 三方案对比 |
| [路线B_SegMAN分割/路线B_fix5k_项目后续步骤.md](路线B_SegMAN分割/路线B_fix5k_项目后续步骤.md) | fix5k 后续步骤 |

---

## 路线 C — 细分类与抓取

| 文档 | 说明 |
| --- | --- |
| [路线C_细分类与抓取/路线C_细分类与抓取实施步骤.md](路线C_细分类与抓取/路线C_细分类与抓取实施步骤.md) | 细分类→ASGrasp→PyBullet |

---

## Trans10K

| 文档 | 说明 |
| --- | --- |
| [Trans10K/Trans10K训练快速开始.md](Trans10K/Trans10K训练快速开始.md) | 训练快速入门 |
| [Trans10K/Trans10K_SegMAN_B_训练与评测结果.md](Trans10K/Trans10K_SegMAN_B_训练与评测结果.md) | 训练评测结果 |

---

## E2E 端到端

| 文档 | 说明 |
| --- | --- |
| [E2E端到端/E2E_segment_and_classify_测试说明.md](E2E端到端/E2E_segment_and_classify_测试说明.md) | E2E 测试说明 |
| [E2E端到端/E2E_性能分析与改进方案.md](E2E端到端/E2E_性能分析与改进方案.md) | 性能分析 |
| [E2E端到端/E2E_实例匹配偏低根因与改进方案.md](E2E端到端/E2E_实例匹配偏低根因与改进方案.md) | match 根因 |
| [E2E端到端/E2E_后续提升方案_F阶段.md](E2E端到端/E2E_后续提升方案_F阶段.md) | F 阶段 M2F 方案 |
| [E2E端到端/F1_匹配后分类优化方案.md](E2E端到端/F1_匹配后分类优化方案.md) | F1 cls 优化 P0/P1 |

---

## 优化 SegMAN match_rate

| 文档 | 说明 |
| --- | --- |
| [优化SegMANmatch_rate/SegMAN_match_rate_提升实施计划.md](优化SegMANmatch_rate/SegMAN_match_rate_提升实施计划.md) | 语义轨 match 提升总计划（§3.1 Phase 1 已完成，§3.2 Phase 2 详细步骤） |
| [优化SegMANmatch_rate/Phase1_后处理优化实验报告.md](优化SegMANmatch_rate/Phase1_后处理优化实验报告.md) | **Phase 1 结题报告**（2026-06-08） |

---

## 其他

| 文档 | 说明 |
| --- | --- |
| [项目总览/项目实施步骤指南.md](项目总览/项目实施步骤指南.md) | 项目总路线图 |

---

## 路径变更说明（2026-06-02）

以下文档由 `SegMAN/` 根目录或 `docs/` 根目录迁入子文件夹：

| 原路径 | 新路径 |
| --- | --- |
| `OpenCLIP_细分类_*.md` | `docs/OpenCLIP细分类/` |
| `路线B_*.md`、`透明物体分割_*.md` | `docs/路线B_SegMAN分割/` |
| `路线C_*.md` | `docs/路线C_细分类与抓取/` |
| `Trans10K*.md` | `docs/Trans10K/` |
| `项目实施步骤指南.md` | `docs/项目总览/` |
| `docs/E2E_*.md`、`docs/F1_*.md` | `docs/E2E端到端/` |
| `docs/SegMAN_OpenCLIP_E2E_*.md` 等 | `docs/交付/` |
