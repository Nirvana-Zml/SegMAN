# Bowl recovery finetune: load fix5k, target cup/background confusion on bowl GT.
# See analyze_bowl_confusion on balanced10k/iter_10000 (~9% bg, ~6% cup on bowl pixels).
_base_ = ['./segman_b_trans10k_lass_balanced.py']

_TRANS10K_CLASS_WEIGHT = [
    1.0,   # background
    1.0,   # box
    1.08,  # bottle
    1.0,   # window
    1.10,  # eyeglass
    1.02,  # freezer
    1.08,  # jar_kettle
    1.10,  # door
    1.0,   # cup (was 1.05 in balanced; reduce cup/bowl competition)
    1.0,   # wall
    1.18,  # bowl
    1.02,  # shelf
]

model = dict(
    backbone=dict(
        lass_cfg=dict(
            enable_stages=[0, 1, 2],
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
                loss_weight=0.15,
                loss_name='loss_dice'),
        ],
        mmscope_cfg=dict(
            enable_bpm=True,
            enable_msbec=True,
            boundary_loss_weight=0.12,
            refine_eta=0.05,
            dilate_kernel=5,
            erode_kernel=5,
        ),
    ),
)

runner = dict(type='IterBasedRunner', max_iters=5000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=5000, metric='mIoU', save_best='mIoU')
