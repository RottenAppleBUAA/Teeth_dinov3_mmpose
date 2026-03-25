# 全景牙片结构化牙根重构

## 项目说明

这个 project 是对原始 `panoramic_teeth` 双任务方案的独立重构版本，不修改也不依赖原 project 的代码。

新的任务定义不再把关键点定位和牙根分割视为两个并列任务，而是显式写成一个结构化串联系统：

- 预测 `root / mesial / distal` 三通道结构图
- 从 `mesial / distal boundary` 响应图中**解码**两条有序 side contour
- 再由 contour **派生** `M_C`、`M_B`、`A`、`D_B`、`D_C`

因此，当前设计落实的是：

1. 显式监督牙根内部、近中边界、远中边界
2. 关键点不再独立自由回归，而是作为结构下游结果生成
3. contour 不再由全局 MLP 直接回归，而是由边界图条件解码，点位天然受边缘线约束

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
- 一个基于结构图的 contour decoder
- 由 contour 派生的 keypoint decoder

关键点由 contour 分支直接重构：

- `M_C = mesial[0]`
- `M_B = mesial[1]`
- `D_B = distal[1]`
- `D_C = distal[0]`
- `A = midpoint(mesial[-1], distal[-1])`

当前前向链路可以概括为：

```text
backbone feature
-> structure logits(root / mesial / distal)
-> boundary-conditioned contour decoding
-> derived keypoints
```

其中 contour decoder 会先预测一组沿牙根深度方向的有序锚点，再在对应行邻域上对 `mesial / distal boundary` 响应做 soft 选择，得到每个 contour 点的 `(x, y)`。这一步是可微的，因此关键点误差可以端到端反向传播到结构图分支。

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

- `loss_root_*` 直接监督牙根主体区域
- `loss_boundary_*` 直接监督近中/远中边界图
- `loss_contour` 监督解码出的 contour 与 GT contour 对齐
- `loss_recon_kpt` 监督由 contour 派生出的 `5` 个关键点
- `loss_attach` 约束 contour 和关键点贴近正确的边界
- `loss_order` 约束 `M_C -> M_B -> A` 和 `D_C -> D_B -> A` 的拓扑顺序
- `loss_apex` 约束 apex 既接近 GT apex，也同时接近两条侧边界

这意味着当前优化不是“mask loss”和“keypoint loss”各自训练各自，而是：

- 结构图损失直接更新结构图分支
- keypoint/contour 几何损失也会通过 contour decoder 回传到结构图分支
- 因此点位监督会反向塑造边界表达，而不是只在末端做独立回归

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

当前默认训练时长为：

- Stage 1: `200` epochs
- Stage 2: `50` epochs

注意：

- 由于 `StructuredContourHead` 已经从早期的 contour 直接回归版升级为“边界图条件解码版”，旧版结构权重与当前 head 不兼容，需要按当前配置重新训练

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
