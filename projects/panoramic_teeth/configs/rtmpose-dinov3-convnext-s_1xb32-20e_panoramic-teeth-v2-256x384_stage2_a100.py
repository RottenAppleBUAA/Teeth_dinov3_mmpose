from pathlib import Path

_base_ = [
    './rtmpose-dinov3-convnext-s_1xb64-60e_panoramic-teeth-v2-256x384_stage1_a100.py'
]

max_epochs = 20
base_lr = 2.5e-5
backend_args = dict(backend='local')
stage1_work_dir = Path(
    './work_dirs/'
    'rtmpose-dinov3-convnext-s_1xb64-60e_panoramic-teeth-v2-256x384_stage1_a100')
work_dir = (
    './work_dirs/'
    'rtmpose-dinov3-convnext-s_1xb32-20e_panoramic-teeth-v2-256x384_stage2_a100')


def _resolve_stage1_best_checkpoint(stage1_dir: Path):
    direct_best = stage1_dir / 'best_NME.pth'
    if direct_best.is_file():
        return direct_best

    candidates = list(stage1_dir.glob('best_NME_epoch_*.pth'))
    if not candidates:
        return None

    def _epoch_key(path: Path) -> int:
        try:
            return int(path.stem.rsplit('_', 1)[-1])
        except ValueError:
            return -1

    return max(candidates, key=_epoch_key)


_stage1_best_checkpoint = _resolve_stage1_best_checkpoint(stage1_work_dir)
load_from = (str(_stage1_best_checkpoint).replace('\\', '/')
             if _stage1_best_checkpoint is not None else None)

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
