_base_ = ['./rtmpose-m_8xb32-60e_panoramic-teeth-v2-256x384.py']

pretrained = (
    'E:/CodeSpace/mmpose/dinov3_weights/'
    'dinov3-convnext-small-pretrain-lvd1689m')
work_dir = (
    './work_dirs/'
    'rtmpose-dinov3-convnext-s_8xb32-60e_panoramic-teeth-v2-256x384_stage1')

model = dict(
    backbone=dict(
        type='DINOv3ConvNextBackbone',
        pretrained=pretrained,
        out_indices=(3, ),
        trainable_stages=(),
        norm_eval=True,
        local_files_only=True),
    head=dict(in_channels=768, in_featuremap_size=(8, 12)),
    mask_head=dict(in_channels=768))
