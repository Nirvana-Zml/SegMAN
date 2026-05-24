# Frozen fix5k delivery recipe (2026-05-23).
# Architecture = segman_b_trans10k_lass.py (unchanged). Only overrides training schedule / lr.
# Checkpoint: outputs/trans10k_lass_mmscope_fix5k/iter_5000.pth
# Test mIoU 80.84%, bowl 80.07%, window 76.27%
_base_ = ['../segman_b_trans10k_lass.py']

runner = dict(type='IterBasedRunner', max_iters=5000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=5000, metric='mIoU', save_best='mIoU')

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0),
            ltab=dict(lr_mult=6.0),
            rsm=dict(lr_mult=6.0),
            bpm=dict(lr_mult=6.0),
            msbec=dict(lr_mult=6.0),
        )))

data = dict(samples_per_gpu=2, workers_per_gpu=2)
