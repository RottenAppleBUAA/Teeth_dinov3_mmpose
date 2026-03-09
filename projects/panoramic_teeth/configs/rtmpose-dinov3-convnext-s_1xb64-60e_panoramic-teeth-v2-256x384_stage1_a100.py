_base_ = ['./rtmpose-dinov3-convnext-s_8xb32-60e_panoramic-teeth-v2-256x384_stage1.py']

base_lr = 5e-4
work_dir = (
    './work_dirs/'
    'rtmpose-dinov3-convnext-s_1xb64-60e_panoramic-teeth-v2-256x384_stage1_a100')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=400),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=30,
        end=60,
        T_max=30,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True)
val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True)
test_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True)
