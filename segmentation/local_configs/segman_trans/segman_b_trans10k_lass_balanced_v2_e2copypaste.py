# Scheme C: Copy-Paste + weak class_weight finetune from v2@6k
# See docs/E2E_实例匹配偏低根因与改进方案.md §6.3.1
_base_ = ['./segman_b_trans10k_lass_balanced_v2.py']

_TRANS10K_CLASS_WEIGHT = [
    1.0,   # background
    1.25,  # box
    1.10,  # bottle
    1.20,  # window
    1.12,  # eyeglass
    1.25,  # freezer
    1.10,  # jar_kettle
    1.25,  # door
    1.0,   # cup
    0.95,  # wall
    1.10,  # bowl
    1.30,  # shelf
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
        mmscope_cfg=dict(boundary_loss_weight=0.18),
    ),
)

optimizer = dict(lr=5e-6)
runner = dict(type='IterBasedRunner', max_iters=3000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')

load_from = 'outputs/trans10k_lass_mmscope_balanced_v2/iter_6000.pth'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 512), ratio_range=(0.5, 2.0)),
    dict(
        type='Trans10KCopyPaste',
        patch_bank='data/trans10k/copypaste_patch_bank.pkl',
        paste_prob=0.5,
        max_paste=2,
        paste_classes=[7, 11, 1, 5, 3],
        scale_range=(0.8, 1.2),
        max_overlap=0.3,
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
)
