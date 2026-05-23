# Phase-1 smoke: LASS backbone + baseline SegMANDecoder (no MMSCopE yet).
_base_ = ['./segman_b_trans10k.py']

model = dict(
    type='EncoderDecoderLASS',
    backbone=dict(
        type='SegMANEncoderLASS_b',
        pretrained='/workspace/segman/pretrained/SegMAN_Encoder_b.pth.tar',
        style='pytorch',
        lass_cfg=dict(
            enable_stages=[0, 1, 2],
            enable_ltab=True,
            enable_rsm=True,
            dilate_kernel=5,
            ltab=dict(beta_init=0.1, alpha_init=1.0, tau_init=0.0),
            rsm=dict(gamma_init=0.5, delta_init=0.5),
        ),
    ),
)

# Smoke / phase-1: shorter run; full training in phase 3 (B7).
runner = dict(type='IterBasedRunner', max_iters=80000)
evaluation = dict(interval=8000, metric='mIoU', save_best='mIoU')
data = dict(samples_per_gpu=2, workers_per_gpu=2)
