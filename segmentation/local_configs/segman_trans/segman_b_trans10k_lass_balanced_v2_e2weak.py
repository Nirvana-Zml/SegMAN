# E2 weak-class finetune (conservative vs P0-1)
# See docs/E2E_性能分析与改进方案.md §7.3
_base_ = ['./segman_b_trans10k_lass_balanced_v2_p0weak.py']

# Stronger boundary for door-wall (balanced_v2 default 0.18)
model = dict(
    decode_head=dict(
        mmscope_cfg=dict(boundary_loss_weight=0.22),
    ),
)

optimizer = dict(lr=5e-6)
runner = dict(type='IterBasedRunner', max_iters=2000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')

load_from = 'outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth'
