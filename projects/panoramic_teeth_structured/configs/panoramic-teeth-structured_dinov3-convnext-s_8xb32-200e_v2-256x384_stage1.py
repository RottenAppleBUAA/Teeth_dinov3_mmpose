_base_ = ['./panoramic-teeth-structured_r50_8xb32-200e_v2-256x384.py']

pretrained = (
    '{{ fileDirname }}/../../../dinov3_weights/'
    'dinov3-convnext-small-pretrain-lvd1689m')
max_epochs = 200
base_lr = 2e-3
work_dir = (
    './work_dirs/'
    'panoramic-teeth-structured_dinov3-convnext-s_8xb32-200e_v2-256x384_stage1')

train_cfg = dict(max_epochs=max_epochs, val_interval=5)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0))

model = dict(
    backbone=dict(
        _delete_=True,
        type='DINOv3ConvNextBackbone',
        pretrained=pretrained,
        out_indices=(3, ),
        trainable_stages=(),
        norm_eval=True,
        local_files_only=True),
    head=dict(in_channels=768))
