# LIDC-IDRI 外部验证数据处理报告

## 当前状态

- 状态：**SAMPLE_EXTRACTION_COMPLETE**
- 数据根目录：`data\external\LIDC-IDRI`
- XML 文件总数：1319
- XML 唯一 SeriesUID：1294
- 去除的重复 XML：25
- 医师级结节标注：20096
- 发现 DICOM series：1
- 与 LUNA16 重叠 series：0
- 非重叠 DICOM series：1
- 已提取 patch：12
- SOP UID 精确对齐 patch：12
- World-Z 回退对齐 patch：0

## 方法约定

1. 使用 DICOM `SeriesInstanceUID` 与 LUNA16 `.mhd` 文件名/metadata 精确去重。
2. 同一 series 的重复 XML 仅选择一个版本；官方 correction/resubmitted 文件优先。
3. XML ROI 中心优先按 `imageSOP_UID` 映射到 DICOM 切片，缺失时才按物理 Z 最近邻回退。
4. 裁剪 3×64×64 patch，并使用 WW=1500、WL=-600 的肺窗归一化。
5. 当前 patch 保留医师级标注。若开展正式外部指标评估，必须另行冻结跨医师共识、
   负候选生成和匹配半径规则，不能直接把医师级重复标注当成独立病例。

## 完整外部验证前仍需满足

- 完整 DICOM 约 124–133 GB，需在容量足够的磁盘通过 TCIA Data Retriever 下载。
- 全量运行后抽检 DICOM 方向矩阵、SOP UID 对齐和 patch 中心。
- 在模型和阈值冻结后，按预注册的共识与负候选协议计算外部指标。
