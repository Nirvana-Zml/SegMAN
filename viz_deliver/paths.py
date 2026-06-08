"""Default data paths relative to SegMAN project root."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / 'output'

F1_PLAN_SUMMARY = ROOT / 'outputs/e2e_improve/f1_plan_summary.json'
E2E_REPORT_A = ROOT / 'outputs/e2e_improve/f1_b1_ref/e2e_metrics_report.json'
E2E_REPORT_B = ROOT / 'outputs/e2e_improve/f1_m2f_e2e/e2e_metrics_report.json'
COVERAGE_GT = ROOT / 'outputs/openclip_classifier/plan_b/coverage_gt/coverage_accuracy.json'
COVERAGE_SEGMAN = ROOT / 'outputs/openclip_classifier/plan_b/coverage_segman/coverage_accuracy.json'
CONFUSION_GT = ROOT / 'outputs/openclip_classifier/deliver_p3/eval_gt_roi/confusion_matrix.json'
PER_CLASS_GT = ROOT / 'outputs/openclip_classifier/deliver_p3/eval_gt_roi/per_class_report.json'
PER_CLASS_SEGMAN = ROOT / 'outputs/openclip_classifier/deliver_p3/eval_segman_roi/per_class_report.json'
