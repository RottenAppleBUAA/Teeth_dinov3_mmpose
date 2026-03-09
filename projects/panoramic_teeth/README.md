# Panoramic Teeth Root Estimation

## Description

This project implements a top-down single-tooth estimator for panoramic dental images.
The model predicts:

- 5 keypoints: `M_C`, `M_B`, `A`, `D_B`, `D_C`
- 1 binary root mask

The current DINOv3 variant uses a shared `DINOv3ConvNextBackbone` backbone and two task heads:

- `RTMCCHead` for 5-point localization
- `RootMaskHead` for root contour segmentation

## Data Preparation

The training config expects the dataset root at:

```text
datasets/586份数据20260116/
```

and the converted v2 annotations at:

```text
datasets/586份数据20260116/annotations_v2/
```

Generate the v2 COCO-style annotations with:

```shell
python tools/dataset_converters/panoramic_teeth_excel_to_coco_v2.py \
    --dataset-root datasets/586份数据20260116 \
    --output-dir datasets/586份数据20260116/annotations_v2
```

The generated files should include:

- `panoramic_teeth_instances_train.json`
- `panoramic_teeth_instances_val.json`
- `panoramic_teeth_instances_test.json`
- `panoramic_teeth_instances_summary.json`

## Training Pipeline

Each single-tooth sample goes through the following top-down pipeline:

```text
LoadImage
-> GetBBoxCenterScale
-> RandomFlip
-> RandomBBoxTransform
-> TopdownAffine
-> GenerateRootMask
-> GenerateTarget
-> PackTeethInputs
```

The role of each stage is:

- `LoadImage`: read the panoramic crop source image
- `GetBBoxCenterScale`: convert the tooth bbox into center/scale form
- `RandomFlip`: apply horizontal augmentation during training
- `RandomBBoxTransform`: apply bbox scale and rotation jitter
- `TopdownAffine`: crop and warp the tooth region to `256 x 384`
- `GenerateRootMask`: rasterize the annotated root polygon into a supervision mask
- `GenerateTarget`: encode the 5 keypoints into SimCC targets
- `PackTeethInputs`: package image, keypoint labels, mask labels, and metainfo

Validation and test use a simplified pipeline:

```text
LoadImage
-> GetBBoxCenterScale
-> TopdownAffine
-> GenerateRootMask
-> PackTeethInputs
```

## Optimization Targets

The model is trained with shared backbone features and two heads.

Training loss is the sum of:

- `loss_kpt`: keypoint SimCC `KLDiscretLoss`
- `loss_mask_bce`: binary cross entropy for the root mask
- `loss_mask_dice`: Dice loss for the root mask

So the model jointly optimizes:

- accurate 5-point tooth landmark localization
- accurate root contour segmentation

Validation reports:

- `NME` for keypoints
- `RootMaskIoUMetric` for the mask

Checkpoint selection currently uses `NME` as the `save_best` target.

## Two-Stage DINOv3 Training

### Stage 1

Config:

```text
projects/panoramic_teeth/configs/rtmpose-dinov3-convnext-s_8xb32-60e_panoramic-teeth-v2-256x384_stage1.py
```

Behavior:

- backbone: `DINOv3ConvNextBackbone`
- feature source: last ConvNeXt stage only
- `head.in_channels = 768`
- `mask_head.in_channels = 768`
- `in_featuremap_size = (8, 12)`
- `trainable_stages = ()`

This stage freezes the whole DINOv3 backbone and trains only the keypoint head and mask head.

Run:

```shell
python tools/train.py \
    projects/panoramic_teeth/configs/rtmpose-dinov3-convnext-s_8xb32-60e_panoramic-teeth-v2-256x384_stage1.py
```

### Stage 2

Config:

```text
projects/panoramic_teeth/configs/rtmpose-dinov3-convnext-s_8xb32-20e_panoramic-teeth-v2-256x384_stage2.py
```

Behavior:

- inherit Stage 1
- `trainable_stages = (3,)`
- load from the best checkpoint in the Stage 1 work directory
- total epochs: `20`
- backbone learning rate multiplier: `0.1`

This stage only unfreezes the last ConvNeXt stage for low-learning-rate fine-tuning.

Run:

```shell
python tools/train.py \
    projects/panoramic_teeth/configs/rtmpose-dinov3-convnext-s_8xb32-20e_panoramic-teeth-v2-256x384_stage2.py
```

## Notes

- The DINOv3 backbone loads local HuggingFace-format weights from:

```text
dinov3_weights/dinov3-convnext-small-pretrain-lvd1689m
```

- Required files in that directory:
  - `config.json`
  - `model.safetensors`

- The current implementation uses local files only and does not fall back to remote downloads.
