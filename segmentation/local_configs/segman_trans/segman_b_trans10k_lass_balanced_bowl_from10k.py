# Mild bowl recovery on balanced10k/iter_10000: keep mIoU/window, lift bowl from 74.31.
# Plan 3: ~9% bg + ~6% cup on bowl GT -> cup 1.0, mild bowl 1.10, low Dice, stages [1,2].
_base_ = ['./segman_b_trans10k_lass_balanced.py']

_TRANS10K_CLASS_WEIGHT = [
    1.0,   # background
    1.0,   # box
    1.10,  # bottle
    1.0,   # window
    1.12,  # eyeglass
    1.02,  # freezer
    1.10,  # jar_kettle
    1.12,  # door
    1.0,   # cup (balanced 1.05 -> 1.0)
    1.0,   # wall
    1.10,  # bowl
    1.05,  # shelf (bowl5k shelf collapse; slight guard)
]

model = dict(
    backbone=dict(
        lass_cfg=dict(
            enable_stages=[1, 2],
            enable_ltab=True,
            enable_rsm=True,
            dilate_kernel=5,
            ltab=dict(beta_init=0.05, alpha_init=1.0, tau_init=0.0),
            rsm=dict(gamma_init=0.3, delta_init=0.3),
        ),
    ),
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
                loss_weight=0.1,
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
    lr=1.5e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0),
            ltab=dict(lr_mult=4.0),
            rsm=dict(lr_mult=4.0),
            bpm=dict(lr_mult=4.0),
            msbec=dict(lr_mult=4.0),
        )))

runner = dict(type='IterBasedRunner', max_iters=3000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=3000, metric='mIoU', save_best='mIoU')
