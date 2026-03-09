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
