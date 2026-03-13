from pathlib import Path

_base_ = [
    './rtmpose-dinov3-convnext-s_1xb64-60e_panoramic-teeth-v2-256x384_stage1_a100.py'
]

max_epochs = 50
base_lr = 2.5e-5
backend_args = dict(backend='local')
stage1_work_dir = Path(
    './work_dirs/'
    'rtmpose-dinov3-convnext-s_1xb64-60e_panoramic-teeth-v2-256x384_stage1_a100')
work_dir = (
    './work_dirs/'
    'rtmpose-dinov3-convnext-s_1xb32-20e_panoramic-teeth-v2-256x384_stage2_a100')
_stage1_best_checkpoint = stage1_work_dir / 'best_NME.pth'
if _stage1_best_checkpoint.is_file():
    load_from = _stage1_best_checkpoint.as_posix()
else:
    load_from = None
    for _candidate in stage1_work_dir.glob('best_NME_epoch_*.pth'):
        try:
            _candidate_epoch = int(_candidate.stem.rsplit('_', 1)[-1])
        except ValueError:
            continue
        if load_from is None or _candidate_epoch > _stage1_best_epoch:
            load_from = _candidate.as_posix()
            _stage1_best_epoch = _candidate_epoch

del Path
del stage1_work_dir
del _stage1_best_checkpoint
if '_candidate' in locals():
    del _candidate
if '_candidate_epoch' in locals():
    del _candidate_epoch
if '_stage1_best_epoch' in locals():
    del _stage1_best_epoch

train_cfg = dict(max_epochs=max_epochs, val_interval=5)

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.85, 1.15],
        rotate_factor=20),
    dict(type='TopdownAffine', input_size=(256, 384)),
    dict(type='GenerateRootMask'),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='SimCCLabel',
            input_size=(256, 384),
            sigma=(5.66, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    dict(type='PackTeethInputs')
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)},
        norm_decay_mult=0,
        bias_decay_mult=0))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=10,
        end=max_epochs,
        T_max=10,
        by_epoch=True,
        convert_to_iter_based=True),
]

model = dict(
    backbone=dict(
        type='DINOv3ConvNextBackbone',
        trainable_stages=(3, ),
        norm_eval=True,
        local_files_only=True))

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline_stage2))
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True)
test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True)
