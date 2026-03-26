custom_imports = dict(
    imports=['projects.panoramic_teeth_structured'],
    allow_failed_imports=False)

_base_ = ['../../../configs/_base_/default_runtime.py']

max_epochs = 200
base_lr = 2e-3
input_size = (192, 512)
contour_points = 16
in_featuremap_size = (6, 16)

train_cfg = dict(max_epochs=max_epochs, val_interval=5)
randomness = dict(seed=21)

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

auto_scale_lr = dict(base_batch_size=256)

codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

point_head_cfg = dict(
    type='RTMCCHead',
    input_size=input_size,
    in_featuremap_size=in_featuremap_size,
    simcc_split_ratio=codec['simcc_split_ratio'],
    final_layer_kernel_size=7,
    gau_cfg=dict(
        hidden_dims=256,
        s=128,
        expansion_factor=2,
        dropout_rate=0.,
        drop_path=0.,
        act_fn='SiLU',
        use_rel_bias=False,
        pos_enc=False),
    loss=dict(
        type='KLDiscretLoss',
        use_target_weight=True,
        beta=10.,
        label_softmax=True),
    decoder=codec)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(type='ResNet', depth=50, out_indices=(3, )),
    head=dict(
        type='AnatomicalRTMCCHead',
        in_channels=2048,
        input_size=input_size,
        contour_points=contour_points,
        feat_channels=256,
        num_convs=2,
        contour_hidden_dim=512,
        point_head_cfg=point_head_cfg,
        root_bce_weight=1.0,
        root_dice_weight=1.0,
        boundary_bce_weight=1.0,
        boundary_dice_weight=1.0,
        contour_weight=2.0,
        contour_attach_weight=0.5,
        mesial_point_weight=1.0,
        apex_point_weight=1.0,
        distal_point_weight=1.0,
        side_attach_weight=0.5,
        side_repel_weight=0.2,
        point_gap_weight=0.2,
        vertical_order_weight=0.2,
        apex_consistency_weight=0.5,
        repel_margin=0.03,
        point_gap_margin=0.02,
        vertical_order_margin=0.01),
    test_cfg=dict(flip_test=False))

dataset_type = 'PanoramicTeethStructuredDataset'
data_mode = 'topdown'
data_root = 'datasets/合并数据/'
metainfo_file = (
    'projects/panoramic_teeth_structured/configs/_base_/datasets/'
    'panoramic_teeth_structured_v2.py')
backend_args = dict(backend='local')

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='ExpandToothBBox', use_tooth_id_templates=True),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=45),
    dict(type='TopdownAffine', input_size=input_size),
    dict(
        type='GenerateAnatomicalToothTargets',
        num_contour_points=contour_points,
        line_width=2,
        point_encoder=codec),
    dict(type='PackAnatomicalToothInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='ExpandToothBBox', use_tooth_id_templates=True),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=input_size),
    dict(
        type='GenerateAnatomicalToothTargets',
        num_contour_points=contour_points,
        line_width=2,
        point_encoder=codec),
    dict(type='PackAnatomicalToothInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=(
            'annotations_v2_good_strict3_pairchecked/'
            'panoramic_teeth_instances_train.json'),
        data_prefix=dict(img=''),
        metainfo=dict(from_file=metainfo_file),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=(
            'annotations_v2_good_strict3_pairchecked/'
            'panoramic_teeth_instances_val.json'),
        data_prefix=dict(img=''),
        metainfo=dict(from_file=metainfo_file),
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=(
            'annotations_v2_good_strict3_pairchecked/'
            'panoramic_teeth_instances_test.json'),
        data_prefix=dict(img=''),
        metainfo=dict(from_file=metainfo_file),
        test_mode=True,
        pipeline=val_pipeline))

default_hooks = dict(
    checkpoint=dict(save_best='NME', rule='less', max_keep_ckpts=1, interval=5))

val_evaluator = [
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 4]),
    dict(type='StructuredFieldMetric', threshold=0.5, prefix='structured'),
]
test_evaluator = val_evaluator
