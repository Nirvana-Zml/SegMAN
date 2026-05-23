# Phase-3 Plan A: LASS encoder + MMSCopE decoder, full 80k on Trans10K-v2.
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
    decode_head=dict(
        type='SegMANDecoderMMSCopE',
        mmscope_cfg=dict(
            enable_bpm=True,
            enable_msbec=True,
            boundary_loss_weight=0.4,
            refine_eta=0.1,
            dilate_kernel=5,
            erode_kernel=5,
        ),
    ),
)

runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=8000, metric='mIoU', save_best='mIoU')
data = dict(samples_per_gpu=2, workers_per_gpu=2)
