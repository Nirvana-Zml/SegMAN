# Decoder-only balanced finetune: baseline SegMANEncoder_b + MMSCopE (no LASS).
# Often preserves per-class IoU better; use if full LASS balanced still regresses.
_base_ = ['./segman_b_trans10k_mmscope_dec_only.py']

_TRANS10K_CLASS_WEIGHT = [
    1.0, 1.0, 1.10, 1.0, 1.12, 1.02, 1.10, 1.12, 1.05, 1.0, 1.0, 1.02,
]

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=_TRANS10K_CLASS_WEIGHT),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                naive_dice=True,
                loss_weight=0.4,
                loss_name='loss_dice'),
        ],
        mmscope_cfg=dict(
            enable_bpm=True,
            enable_msbec=True,
            boundary_loss_weight=0.15,
            refine_eta=0.05,
            dilate_kernel=5,
            erode_kernel=5,
        ),
    ),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0),
            bpm=dict(lr_mult=6.0),
            msbec=dict(lr_mult=6.0),
        )))

runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=10000, metric='mIoU', save_best='mIoU')
data = dict(samples_per_gpu=2, workers_per_gpu=2)
