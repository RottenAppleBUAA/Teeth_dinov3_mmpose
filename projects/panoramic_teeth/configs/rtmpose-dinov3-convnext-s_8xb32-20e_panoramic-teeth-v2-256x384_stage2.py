from pathlib import Path

_base_ = [
    './rtmpose-dinov3-convnext-s_8xb32-60e_panoramic-teeth-v2-256x384_stage1.py'
]

max_epochs = 20
base_lr = 2e-4
stage1_work_dir = Path(
    './work_dirs/'
    'rtmpose-dinov3-convnext-s_8xb32-60e_panoramic-teeth-v2-256x384_stage1')
work_dir = (
    './work_dirs/'
    'rtmpose-dinov3-convnext-s_8xb32-20e_panoramic-teeth-v2-256x384_stage2')
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
        end=200),
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
