# 全景片牙齿 Landmark 任务改造方案

## Summary
在现有 MMPose 上复用 `TopdownPoseEstimator`，但把任务定义成“每张全景片 1 个实例、固定输出 32 牙位的整图 landmark”。  
主干使用本地适配的 `DINOv3`，预测头使用 `HeatmapHead`。  
数据先从 Excel 转成 COCO 风格单实例标注：每图 1 条 annotation，`bbox` 为整张图，`keypoints` 为固定拓扑展开后的整图点位。

固定输出拓扑定义为：
- 牙位集合：`11-18, 21-28, 31-38, 41-48`
- 每牙输出 12 个点：`M_side` 5 个主点 + `M_AL` 1 个辅助点 + `D_side` 5 个主点 + `D_AL` 1 个辅助点
- 总输出点数：`32 x 12 = 384`
- 缺失牙或该侧无标注时，坐标置零，`visibility=0`

## Key Changes
### 1. 数据转换
新增一个 Excel-to-COCO 转换脚本，输入 `586` 份 Excel、原图和描点图，输出一个全景片单实例数据集。
转换规则固定如下：
- 按表头名解析，兼容当前 3 种 Excel 结构，不按固定列号写死
- 每张图聚合成 1 条 annotation
- 图像名按 Excel 文件号映射到原图，例如 `excel1... -> 原图1...`
- `bbox` 直接使用整张图范围 `[0, 0, width, height]`
- 对每颗牙，从 `测量信息` 中抽取 `M/D` 两侧点列
- 原始点列先做“相邻重复点折叠”
- 折叠后前 5 个点作为该侧 5 个主点
- `AL` 辅助点规则：
  - 如果存在与第 2 主点不重合的牙龈分界点，则写入 `*_AL`
  - 如果与第 2 主点重合，则 `*_AL` 置零并 `visibility=0`
- 最终按固定顺序展开成 384 个点：
  - 每牙顺序：`M1..M5, M_AL, D1..D5, D_AL`
  - 牙位顺序固定为 `11,12,...,18,21,...,48`
- 同时导出一份可视化校验结果，把转换后的整图点重新画回原图，抽样核对

### 2. Dataset / Metainfo
不新写复杂 dataset 类，直接使用转换后的 COCO 风格整图数据。
新增一个专用 metainfo 文件，定义：
- 384 个 keypoint 名称
- 每个 keypoint 的牙位、左右侧、点序号
- `swap=''`，首版不启用水平翻转
- `joint_weights` 默认全 1；`AL` 点可设较低权重，如 `0.5`

评估采用：
- `NME`，`norm_mode='use_norm_item'`，归一化因子用整图 `bbox_size`
- `EPE`
- 可选增加每牙位平均误差统计，但不作为首版阻塞项

### 3. DINOv3 Backbone 适配
新增本地 `DINOv3` 适配 backbone，接口形态固定为：
- `type='DinoV3Backbone'`
- `checkpoint='...'`
- `img_size=(1024, 512)`
- 输出为 2D `featmap`，供 `HeatmapHead` 直接消费
- 隐藏 `cls token`，只保留 patch 特征
- 若原始 DINOv3 输出是 token 序列，则在适配器里 reshape 成 `[B, C, H/16, W/16]`

默认训练输入分辨率定为 `1024x512`，原因是：
- 保持全景片约 2:1 宽高比
- 对 `DINOv3` patch 特征和 384 通道 heatmap 的显存更可控
- 便于输出 `256x128` heatmap

### 4. 模型与训练配置
模型保持单头 heatmap 方案，不拆多任务分类头。
配置固定为：
- `TopdownPoseEstimator`
- `backbone=DinoV3Backbone`
- `head=HeatmapHead`
- `out_channels=384`
- `decoder=UDPHeatmap`
- `input_size=(1024, 512)`
- `heatmap_size=(256, 128)`
- `loss=KeypointMSELoss(use_target_weight=True)`

数据增强固定为：
- 保留：亮度/对比度、轻微旋转、轻微缩放
- 不启用：水平翻转
  - 原因：牙位语义强绑定左右象限，首版避免引入复杂 swap 规则
- 不做半身/检测框扰动
  - 原因：任务是整图单实例

训练默认：
- AMP 打开
- 小 batch，默认 `batch_size=2`
- `AdamW`
- 先训主干+头全量微调
- checkpoint 路径由配置显式提供，不做在线下载

## Test Plan
1. 转换脚本单测覆盖 3 种 Excel 表头格式。  
2. 单测覆盖点列长度 `5/6/7`、缺失牙、仅单侧标注、`AL` 重合和不重合两种情况。  
3. 校验转换后单张样本一定输出 `384` 点，且 visibility mask 长度一致。  
4. 校验 `DinoV3Backbone` 对 `1024x512` 输入输出固定形状的 featmap。  
5. 跑一次模型 `forward/loss/predict` smoke test，确认 `HeatmapHead(out_channels=384)` 正常。  
6. 对抽样原图生成 overlay，可视化确认整图点位和描点图一致。  
7. 用极小子集做 overfit，确认 loss 能明显下降。  

## Assumptions
- `PointList` 按你的说明视为整张全景片坐标，首版不再引入局部框坐标反算。
- 首版只做 landmark，不并入牙周炎等级分类。
- `AL` 作为辅助点进入固定拓扑，而不是另起一个标量分支。
- 若没有现成官方划分，默认按图像做固定随机划分 `8/1/1`，`seed=42`。
- 如果可视化校验发现 `PointList` 与原图坐标不一致，则优先修正数据转换，不改模型主方案。
