# 全景牙片牙根估计

## 项目说明

这个项目实现了一个面向全景牙片的单牙 top-down 估计模型。
模型同时预测：

- 5 个关键点：`M_C`、`M_B`、`A`、`D_B`、`D_C`
- 1 张二值牙根 mask

当前的 DINOv3 版本使用共享的 `DINOv3ConvNextBackbone` 作为 backbone，并接两个任务头：

- `RTMCCHead`：用于 5 点定位
- `RootMaskHead`：用于牙根轮廓分割

## 数据准备

当前训练配置默认使用以下数据根目录：

```text
datasets/586份数据20260116/
```

并要求 v2 标注位于：

```text
datasets/586份数据20260116/annotations_v2/
```

使用下面的命令生成 v2 COCO 风格标注：

```shell
python tools/dataset_converters/panoramic_teeth_excel_to_coco_v2.py \
    --dataset-root datasets/586份数据20260116 \
    --output-dir datasets/586份数据20260116/annotations_v2
```

生成后至少应包含以下文件：

- `panoramic_teeth_instances_train.json`
- `panoramic_teeth_instances_val.json`
- `panoramic_teeth_instances_test.json`
- `panoramic_teeth_instances_summary.json`

## 训练 Pipeline

每个单牙样本在训练时经过以下 top-down 流程：

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

各阶段作用如下：

- `LoadImage`：读取原始牙片图像
- `GetBBoxCenterScale`：将单牙 bbox 转成 center/scale 表示
- `RandomFlip`：训练时进行水平翻转增强
- `RandomBBoxTransform`：对 bbox 做缩放和旋转扰动
- `TopdownAffine`：将单牙区域裁剪并仿射变换到 `256 x 384`
- `GenerateRootMask`：将标注中的牙根多边形栅格化成监督 mask
- `GenerateTarget`：将 5 个关键点编码成 SimCC 监督目标
- `PackTeethInputs`：打包图像、关键点标签、mask 标签和元信息

验证和测试使用简化后的流程：

```text
LoadImage
-> GetBBoxCenterScale
-> TopdownAffine
-> GenerateRootMask
-> PackTeethInputs
```

## 优化目标

当前模型使用共享 backbone 特征，并同时训练两个 head。

训练总 loss 由以下三部分组成：

- `loss_kpt`：关键点 SimCC 的 `KLDiscretLoss`
- `loss_mask_bce`：牙根 mask 的二值交叉熵
- `loss_mask_dice`：牙根 mask 的 Dice loss

因此模型的联合优化目标是：

- 提高 5 个关键点定位精度
- 提高牙根轮廓分割精度

验证阶段会同时输出：

- 关键点的 `NME`
- mask 的 `RootMaskIoUMetric`

当前 best checkpoint 的选择指标是 `NME`。

## 两阶段 DINOv3 训练

### Stage 1

配置文件：

```text
projects/panoramic_teeth/configs/rtmpose-dinov3-convnext-s_8xb32-60e_panoramic-teeth-v2-256x384_stage1.py
```

配置特点：

- backbone：`DINOv3ConvNextBackbone`
- 仅使用最后一个 ConvNeXt stage 的特征
- `head.in_channels = 768`
- `mask_head.in_channels = 768`
- `in_featuremap_size = (8, 12)`
- `trainable_stages = ()`

这一阶段会冻结整个 DINOv3 backbone，只训练关键点头和 mask 头。

训练命令：

```shell
python tools/train.py \
    projects/panoramic_teeth/configs/rtmpose-dinov3-convnext-s_8xb32-60e_panoramic-teeth-v2-256x384_stage1.py
```

### Stage 2

配置文件：

```text
projects/panoramic_teeth/configs/rtmpose-dinov3-convnext-s_8xb32-20e_panoramic-teeth-v2-256x384_stage2.py
```

配置特点：

- 继承 Stage 1
- `trainable_stages = (3,)`
- 从 Stage 1 工作目录中的最佳 checkpoint 继续训练
- 总 epoch 数为 `20`
- backbone 学习率倍率为 `0.1`

这一阶段只解冻最后一个 ConvNeXt stage，并以较低学习率做微调。

训练命令：

```shell
python tools/train.py \
    projects/panoramic_teeth/configs/rtmpose-dinov3-convnext-s_8xb32-20e_panoramic-teeth-v2-256x384_stage2.py
```

## 说明

- DINOv3 backbone 从本地 HuggingFace 格式权重目录加载：

```text
dinov3_weights/dinov3-convnext-small-pretrain-lvd1689m
```

- 该目录下至少需要：
  - `config.json`
  - `model.safetensors`

- 当前实现固定使用本地文件，不会回退到网络下载。
