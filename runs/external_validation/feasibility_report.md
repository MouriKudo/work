# LIDC-IDRI 外部验证可行性报告

## 当前状态

- 状态：**DATA_NOT_PRESENT**
- 数据根目录：`D:\luna16-work\data\external\LIDC-IDRI`
- 发现 DICOM series：0
- 解析 XML 文件：0
- 放射科医师级结节标注：0
- 与 LUNA16 重叠 series：0
- 可作为非重叠外部病例的 series：0

## 方法约定

1. 使用 DICOM `SeriesInstanceUID` 与 LUNA16 `.mhd` 文件名精确去重。
2. XML 的 ROI 轮廓中心映射到最近物理 Z 层，再裁剪 3×64×64 patch。
3. 使用与 LUNA16 一致的肺窗（WW=1500、WL=-600）归一化。
4. 不同阅片医师的标注当前独立保留；正式报告前需声明共识合并规则。
5. XML 只直接提供结节标注。若需计算外部 AUC/F1/FROC，还必须定义负候选生成器、匹配半径及完整候选评估协议。

## 正式外部验证前检查清单

- [ ] 下载并校验完整 LIDC-IDRI DICOM 与 XML。
- [ ] 确认非重叠病例清单不含任何 LUNA16 series。
- [ ] 冻结阅片医师共识/恶性评分到二分类标签的规则。
- [ ] 生成不依赖测试标签的外部负候选。
- [ ] 抽检 DICOM 方向矩阵、XML 坐标和 patch 中心对齐。
- [ ] 冻结模型与阈值后一次性执行外部评估。
