# Route B balanced finetune: keep fix5k gains (window/mIoU) while lifting regressed classes.
# Load baseline iter_80000, train ~10k @ lr=2e-5; mild boundary + class-weighted CE + Dice.
_base_ = ['./segman_b_trans10k_lass.py']

# Mild boost for classes that often drop under strong boundary/LASS (fix5k post-mortem).
_TRANS10K_CLASS_WEIGHT = [
    1.0,   # background
    1.0,   # box
    1.10,  # bottle
    1.0,   # window (already strong; avoid over-weight)
    1.12,  # eyeglass
    1.02,  # freezer
    1.10,  # jar_kettle
    1.12,  # door
    1.05,  # cup
    1.0,   # wall
    1.0,   # bowl
    1.02,  # shelf
]

model = dict(
    backbone=dict(
        lass_cfg=dict(
            # Skip stage0: less disruption to low-level features (bottle/cup/eyeglass).
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
            ltab=dict(lr_mult=6.0),
            rsm=dict(lr_mult=6.0),
            bpm=dict(lr_mult=6.0),
            msbec=dict(lr_mult=6.0),
        )))

runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=10000, metric='mIoU', save_best='mIoU')
data = dict(samples_per_gpu=2, workers_per_gpu=2)
