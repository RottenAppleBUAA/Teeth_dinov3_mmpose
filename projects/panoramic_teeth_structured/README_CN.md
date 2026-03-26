# 全景牙片结构化牙根重构

## 项目说明

这个 project 是对原始 `panoramic_teeth` 双任务方案的独立重构版本。

它不再把关键点定位和牙根分割视为两个弱耦合任务，而是把任务改写为：

- 预测 `root mask`
- 预测 `mesial / distal` 两条有序边界
- 预测两条有序 side contour polyline
- 由 contour **重构** `M_C`、`M_B`、`A`、`D_B`、`D_C`

因此新的训练设计同时落实三件事：

1. 显式监督牙根内部、近中边界、远中边界
2. 增加几何一致性损失，约束关键点必须附着在正确边界上，并保持 `2 + 1 + 2` 的拓扑顺序
3. 用 contour reconstruction 取代独立 keypoint head，使关键点成为结构重建结果，而不是自由回归结果

## 数据格式

新 project 直接读取现有 v2 COCO 标注，但要求 annotation 中包含：

- `segmentation`
- `side_contours.M`
- `side_contours.D`
- `apex_midpoint`

当前实现默认使用：

```text
datasets/合并数据/
```

并默认读取：

```text
datasets/合并数据/annotations_v2_good_strict3_pairchecked/
```

## 模型设计

当前模型使用共享 backbone 和一个 `StructuredContourHead`：

- `root / mesial / distal` 三通道结构图分支
- `mesial / distal` 两条 polyline 的 contour 回归分支

关键点由 contour 分支直接重构：

- `M_C = mesial[0]`
- `M_B = mesial[1]`
- `D_B = distal[1]`
- `D_C = distal[0]`
- `A = midpoint(mesial[-1], distal[-1])`

## 损失设计

训练总 loss 由以下部分组成：

- `loss_root_bce`
- `loss_root_dice`
- `loss_boundary_bce`
- `loss_boundary_dice`
- `loss_contour`
- `loss_recon_kpt`
- `loss_attach`
- `loss_order`
- `loss_apex`

其中：

- `loss_attach` 约束预测 contour 和关键点贴近对应的近中/远中 GT 边界
- `loss_order` 约束 `M_C -> M_B -> A` 和 `D_C -> D_B -> A` 的几何顺序
- `loss_apex` 约束 apex 既接近 GT apex，也同时接近近中/远中两条边界

## 配置

基础配置：

```text
projects/panoramic_teeth_structured/configs/panoramic-teeth-structured_r50_8xb32-200e_v2-256x384.py
```

DINOv3 两阶段配置：

```text
projects/panoramic_teeth_structured/configs/panoramic-teeth-structured_dinov3-convnext-s_8xb32-200e_v2-256x384_stage1.py
projects/panoramic_teeth_structured/configs/panoramic-teeth-structured_dinov3-convnext-s_8xb32-50e_v2-256x384_stage2.py
```

解剖点主头实验配置：

```text
projects/panoramic_teeth_structured/configs/panoramic-teeth-anatomical_r50_8xb32-200e_v2-192x512.py
projects/panoramic_teeth_structured/configs/panoramic-teeth-anatomical_dinov3-convnext-s_8xb32-200e_v2-192x512_stage1.py
projects/panoramic_teeth_structured/configs/panoramic-teeth-anatomical_dinov3-convnext-s_8xb32-50e_v2-192x512_stage2.py
```

这组 anatomy 配置采用：

- `RTMCC / SimCC` 作为关键点主任务
- `A`、`M_C/M_B`、`D_B/D_C` 三个解剖子头
- `root / mesial / distal` 结构图和 contour 作为辅助语义分支
- 基于现有 keypoint 和 side contour 自动构造的 anatomy pseudo targets

当前默认训练时长为：

- Stage 1: `200` epochs
- Stage 2: `50` epochs

## 评估

验证阶段输出：

- 关键点 `NME`
- `StructuredFieldMetric`

其中 `StructuredFieldMetric` 会统计：

- `root_mIoU`
- `root_mDice`
- `mesial_mDice`
- `distal_mDice`
- `boundary_mDice`
