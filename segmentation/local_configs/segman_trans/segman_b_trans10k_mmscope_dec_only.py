# Phase-2.6a: baseline SegMANEncoder_b + SegMANDecoderMMSCopE (decoder-only smoke).
_base_ = ['./segman_b_trans10k.py']

model = dict(
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
evaluation = dict(interval=8000, metric='mIoU', save_best='mIoU')
data = dict(samples_per_gpu=2, workers_per_gpu=2)
