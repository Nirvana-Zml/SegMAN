# P0 weak-class finetune: from balanced_v2 @ iter_6000
# See OpenCLIP_细分类_未达80%原因与优化方案.md §5.2 P0-1
_base_ = ['./segman_b_trans10k_lass_balanced_v2.py']

_TRANS10K_CLASS_WEIGHT = [
    1.0,   # background
    1.15,  # box
    1.10,  # bottle
    1.08,  # window
    1.12,  # eyeglass
    1.15,  # freezer
    1.10,  # jar_kettle
    1.15,  # door
    1.0,   # cup
    0.95,  # wall
    1.12,  # bowl
    1.20,  # shelf
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
                loss_weight=0.18,
                loss_name='loss_dice'),
            dict(
                type='BowlAntiCupLoss',
                bowl_class_index=10,
                cup_class_index=8,
                loss_weight=0.25,
                loss_name='loss_bowl_ac',
                ignore_index=255),
        ],
    ),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-5,
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
            layers_0=dict(lr_mult=0.5),
            layers_1=dict(lr_mult=0.5),
        )))

runner = dict(type='IterBasedRunner', max_iters=4000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
data = dict(samples_per_gpu=2, workers_per_gpu=2)
