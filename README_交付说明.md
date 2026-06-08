# TransGrasp 双轨交付 V1.0.0

透明物体 **11 类** 感知与细分类；**一套软件、两种模式**。

| 模式 | 入口 | 场景 |
| --- | --- | --- |
| **A 语义** | `python app/run_semantic_e2e.py` | SegMAN 全图语义 + 细分类 |
| **B 抓取** | `python app/run_grasp_e2e.py` | M2F 实例 + 细分类（**推荐抓取**） |

完整说明：`docs/交付/SegMAN_OpenCLIP_E2E_交付路线.md`  
文档索引：`docs/README.md`  
Manifest：`outputs/e2e_improve/deliver_dual_track_manifest.json`  
Phase 1 优化记录：`docs/优化SegMANmatch_rate/Phase1_后处理优化实验报告.md`（语义轨后处理 sweep，deploy 仍 59.16% B1）

## 快速开始

```bash
# 模式 A（conda: segman）
python app/run_semantic_e2e.py --image your.jpg --out-dir outputs/demo_semantic

# 模式 B（conda: segman_mmdet）
python app/run_grasp_e2e.py --image your.jpg --out-dir outputs/demo_grasp
```

## 验收

```bash
bash scripts/run_deliver_semantic_e2e.sh
bash scripts/run_deliver_grasp_e2e.sh
```
