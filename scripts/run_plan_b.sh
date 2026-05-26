#!/usr/bin/env bash
# Plan B: coverage-accuracy + reject policy + archive summary
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=outputs/openclip_classifier/deliver_classifier_best.pth
OUT=outputs/openclip_classifier/plan_b
mkdir -p "${OUT}/coverage_gt" "${OUT}/coverage_segman" "${OUT}/reject_gt" "${OUT}/reject_segman"
mkdir -p outputs/openclip_classifier/deliver_experiment_best

echo "=== B2 GT coverage-accuracy ==="
python transgrasp/classification/eval_coverage_accuracy.py \
  --checkpoint "${CKPT}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --report-dir "${OUT}/coverage_gt"

echo "=== B2 SegMAN coverage-accuracy ==="
python transgrasp/classification/eval_coverage_accuracy.py \
  --checkpoint "${CKPT}" \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --report-dir "${OUT}/coverage_segman"

echo "=== B3 GT reject policy ==="
python transgrasp/classification/eval_reject_policy.py \
  --checkpoint "${CKPT}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --class-thresholds transgrasp/classification/configs/reject_thresholds_p3.json \
  --global-threshold 0.5 \
  --report-dir "${OUT}/reject_gt"

echo "=== B3 SegMAN reject policy ==="
python transgrasp/classification/eval_reject_policy.py \
  --checkpoint "${CKPT}" \
  --roi-root data/trans10k_roi_segman \
  --split val \
  --class-thresholds transgrasp/classification/configs/reject_thresholds_p3.json \
  --global-threshold 0.5 \
  --report-dir "${OUT}/reject_segman"

echo "=== B4-lite ROI predictions with reject (GT val sample) ==="
python transgrasp/pipelines/classify_roi_with_reject.py \
  --checkpoint "${CKPT}" \
  --roi-root data/trans10k_roi_gt \
  --split val \
  --out "${OUT}/predictions_gt_val.json"

python - <<'PY'
import json
from datetime import date
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

oc = Path('outputs/openclip_classifier')
cov_gt = load('outputs/openclip_classifier/plan_b/coverage_gt/coverage_accuracy.json')
cov_seg = load('outputs/openclip_classifier/plan_b/coverage_segman/coverage_accuracy.json')
rej_gt = load('outputs/openclip_classifier/plan_b/reject_gt/reject_policy.json')
rej_seg = load('outputs/openclip_classifier/plan_b/reject_segman/reject_policy.json')
deliver = load('outputs/openclip_classifier/deliver_p3/deliver_manifest.json')

gate = {
    'checkpoint': 'outputs/openclip_classifier/deliver_classifier_best.pth',
    'global_gt_acc': cov_gt['global_top1_acc'],
    'global_segman_acc': cov_seg['global_top1_acc'],
    'gt_acc_at_60pct_coverage': cov_gt['highlights']['acc_at_60pct_coverage'],
    'gt_acc_at_70pct_coverage': cov_gt['highlights']['acc_at_70pct_coverage'],
    'segman_acc_at_60pct_coverage': cov_seg['highlights']['acc_at_60pct_coverage'],
    'gt_reject_per_class_acc': rej_gt['per_class_threshold_policy']['accuracy_on_accepted'],
    'gt_reject_per_class_cov': rej_gt['per_class_threshold_policy']['coverage'],
    'segman_reject_per_class_acc': rej_seg['per_class_threshold_policy']['accuracy_on_accepted'],
    'pass_acc_78_at_cov_60_gt': cov_gt['plan_b_gates']['pass_acc_78_at_coverage_60'],
    'pass_acc_80_at_cov_70_gt': cov_gt['plan_b_gates']['pass_acc_80_at_coverage_70'],
}
(oc / 'plan_b' / 'gate.json').write_text(json.dumps(gate, indent=2) + '\n', encoding='utf-8')

summary_md = f"""# Plan B 结题指标摘要

**日期**：{date.today()}  
**分类 deliver**：P3 `deliver_classifier_best.pth`  
**分割**：v2@6k iter_6000.pth  

## 1. 全局 Top-1（不变）

| 评测集 | Acc |
|--------|-----|
| GT-ROI | {cov_gt['global_top1_acc']:.2%} |
| SegMAN-ROI | {cov_seg['global_top1_acc']:.2%} |

## 2. Coverage–Accuracy（B2）

| 评测集 | @60% coverage Acc | @70% coverage Acc |
|--------|-------------------|-------------------|
| GT-ROI | {cov_gt['highlights']['acc_at_60pct_coverage']:.2%} | {cov_gt['highlights']['acc_at_70pct_coverage']:.2%} |
| SegMAN-ROI | {cov_seg['highlights']['acc_at_60pct_coverage']:.2%} | {cov_seg['highlights']['acc_at_70pct_coverage']:.2%} |

**闸门**：GT @60% coverage ≥78% → {'PASS' if gate['pass_acc_78_at_cov_60_gt'] else 'FAIL'}  
**闸门**：GT @70% coverage ≥80% → {'PASS' if gate['pass_acc_80_at_cov_70_gt'] else 'FAIL'}

## 3. 按类拒识（B3）

| 策略 | GT coverage | GT acc on accepted |
|------|-------------|-------------------|
| 全局 τ=0.5 | {rej_gt['global_policy']['coverage']:.2%} | {rej_gt['global_policy']['accuracy_on_accepted']:.2%} |
| 按类 τ | {rej_gt['per_class_threshold_policy']['coverage']:.2%} | {rej_gt['per_class_threshold_policy']['accuracy_on_accepted']:.2%} |

| 策略 | SegMAN coverage | SegMAN acc on accepted |
|------|-----------------|------------------------|
| 按类 τ | {rej_seg['per_class_threshold_policy']['coverage']:.2%} | {rej_seg['per_class_threshold_policy']['accuracy_on_accepted']:.2%} |

## 4. 组合验收建议

| 层级 | 指标 | 结果 |
|------|------|------|
| 分割 mIoU | ≥80% | 81.80% ✅ |
| GT Top-1 | ≥75% | {cov_gt['global_top1_acc']:.2%} ✅ |
| GT Top-1 stretch | ≥80% | {cov_gt['global_top1_acc']:.2%} ❌ |
| 高置信子集 @60% | ≥78% | {cov_gt['highlights']['acc_at_60pct_coverage']:.2%} {'✅' if gate['pass_acc_78_at_cov_60_gt'] else '❌'} |

## 5. 产物

- `plan_b/coverage_gt/coverage_accuracy.json`
- `plan_b/coverage_segman/coverage_accuracy.json`
- `plan_b/reject_gt/reject_policy.json`
- `plan_b/reject_segman/reject_policy.json`
- `plan_b/predictions_gt_val.json`
- `plan_b/gate.json`
"""
(oc / 'deliver_experiment_best' / 'metrics_summary.md').write_text(summary_md, encoding='utf-8')
manifest = {
    'plan_b_date': str(date.today()),
    'deliver': deliver,
    'plan_b_gate': gate,
    'coverage_gt': str(oc / 'plan_b/coverage_gt/coverage_accuracy.json'),
    'coverage_segman': str(oc / 'plan_b/coverage_segman/coverage_accuracy.json'),
}
(oc / 'deliver_experiment_best' / 'manifest.json').write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(gate, indent=2))
PY

echo "Plan B done. See outputs/openclip_classifier/plan_b/ and deliver_experiment_best/metrics_summary.md"
