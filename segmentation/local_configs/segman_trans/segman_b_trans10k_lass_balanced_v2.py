# Route B balanced-v2: lift bowl on top of balanced recipe (vs balanced10k iter_10000).
# Changes vs v1 balanced: LASS stages [0,1,2] weak, lower Dice, cup=1.0, bowl=1.10, BowlAntiCupLoss.
# Does NOT modify segman_b_trans10k_lass.py (fix5k recipe unchanged).
_base_ = ['./segman_b_trans10k_lass.py']

_TRANS10K_CLASS_WEIGHT = [
    1.0,   # background
    1.0,   # box
    1.10,  # bottle
    1.0,   # window
    1.12,  # eyeglass
    1.02,  # freezer
    1.10,  # jar_kettle
    1.12,  # door
    1.0,   # cup (v1 was 1.05)
    1.0,   # wall
    1.10,  # bowl (v1 was 1.0)
    1.05,  # shelf
]

model = dict(
    backbone=dict(
        lass_cfg=dict(
            enable_stages=[0, 1, 2],
            enable_ltab=True,
            enable_rsm=True,
            dilate_kernel=5,
            # Weaker LASS init than fix5k (0.1/0.5), stronger than balanced v1 (0.05/0.3)
            ltab=dict(beta_init=0.06, alpha_init=1.0, tau_init=0.0),
            rsm=dict(gamma_init=0.35, delta_init=0.35),
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
                loss_weight=0.15,
                loss_name='loss_dice'),
            dict(
                type='BowlAntiCupLoss',
                bowl_class_index=10,
                cup_class_index=8,
                loss_weight=0.25,
                loss_name='loss_bowl_ac',
                ignore_index=255),
        ],
        mmscope_cfg=dict(
            enable_bpm=True,
            enable_msbec=True,
            boundary_loss_weight=0.18,
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
            ltab=dict(lr_mult=5.0),
            rsm=dict(lr_mult=5.0),
            bpm=dict(lr_mult=5.0),
            msbec=dict(lr_mult=5.0),
            # Shallow encoder blocks: smaller lr (param name substring match)
            layers_0=dict(lr_mult=0.5),
            layers_1=dict(lr_mult=0.5),
        )))

runner = dict(type='IterBasedRunner', max_iters=8000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=8000, metric='mIoU', save_best='mIoU')
data = dict(samples_per_gpu=2, workers_per_gpu=2)
