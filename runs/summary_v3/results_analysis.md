# 肺结节分类实验结果分析

> 本文档由 `src/export_tables.py` 从实际 CSV/JSON 自动生成。所有结论仅描述当前保存的实验结果。

## 1. 主实验结果

- 平均 AUC 最佳方法为 **ResNet18（强增强）**：0.99625 ± 0.00125。
- 平均 PR-AUC 最佳方法为 **ResNet18（强增强）**：0.98814 ± 0.00244。
- 平均 F1 最佳方法为 **PBIP-Lite**：0.93569 ± 0.02123。

PBIP-Lite 相对强增强 ResNet18 的平均 F1 变化为 +0.01196，平均 AUC 变化为 -0.00125。该结果说明当前实验中原型融合更偏向改善阈值相关的分类平衡，而最高平均 AUC 仍由强增强 ResNet18 获得。

## 2. 消融实验

- K 值消融中，AUC 最佳为 **K=3**（0.99542），F1 最佳为 **K=5**（0.94949）。
- β 消融中，AUC 最佳为 **β=0.10**（0.99583），F1 最佳为 **β=0.05**（0.94301）。
- 移除原型对比损失：AUC=0.99501，PR-AUC=0.98759，F1=0.91787。
- 移除原型 logit：AUC=0.99468，PR-AUC=0.98738，F1=0.93750。
- 消融结果均为 seed 0，适合描述组件趋势，不应替代三种子主实验的稳定性结论。

## 3. 图像退化鲁棒性

- 六类 Level 5 退化的宏平均 AUC 最佳为 **ResNet18（强增强）**：0.97380，相对干净数据平均下降 0.02114。
- 六类 Level 5 退化的宏平均 F1 最佳为 **PBIP + 原型对比损失**：0.86759，相对干净数据平均下降 0.06241。
- ResNet18（无增强）：AUC 下降最大退化为低对比度（下降 0.07483）；F1 下降最大退化为下采样（下降 0.25530）。
- ResNet18（强增强）：AUC 下降最大退化为低对比度（下降 0.08217）；F1 下降最大退化为下采样（下降 0.15396）。
- PBIP-Lite：AUC 下降最大退化为低对比度（下降 0.08488）；F1 下降最大退化为低对比度（下降 0.16913）。
- PBIP + 原型对比损失：AUC 下降最大退化为低对比度（下降 0.08923）；F1 下降最大退化为低对比度（下降 0.16913）。
- 所有退化评估均沿用干净验证集冻结阈值，因此 F1 变化同时反映判别能力和概率校准变化。

## 4. Grad-CAM 解释性实验

- ResNet18（强增强） 代表样本数：TP=3、FP=3、FN=2、TN=3。
- PBIP + 原型对比损失 代表样本数：TP=3、FP=3、FN=3、TN=3。
- 热图由阳性 logit 对 ResNet18 最后一个残差块反向传播得到，叠加在三层输入的中心层上。
- 当前实验没有像素级病灶分割真值，因此本表只报告样本覆盖和预测概率，不能把热区视觉一致性表述为定量定位精度。

## 5. 外部/固定测试

- 本次实际评估数据源为 **luna16_fixed_test**。LIDC-IDRI 自动回退原因：LIDC-IDRI 不满足二分类评估要求，实际类别=[1]，必须同时包含 0/1；LIDC-IDRI 仅有 1 个 CT，无法执行 CT 簇 Bootstrap。
- PBIP-Lite：Acc=0.97384±0.00922，F1=0.93569±0.02123，AUC=0.99500±0.00008。
- ResNet18（强增强）：Acc=0.96848±0.00506，F1=0.92373±0.01068，AUC=0.99625±0.00125。
- 三类检验在五项指标上的最小原始 p 值为 0.22344，p<0.05 的记录数为 0。由于仅有 3 个随机种子，未显著不等同于两种方法性能等效。
- 95% CI 使用 seriesuid 为单位的 1,000 次 CT 簇 Bootstrap；Precision、Recall 和 F1 使用各模型验证集冻结阈值。

## 6. 数据来源与解释边界

- `D:\luna16-work\runs\summary_v3\main_results.csv`
- `D:\luna16-work\runs\ablations\ablation_results.csv`
- `D:\luna16-work\runs\robustness\robustness_detailed.csv`
- `D:\luna16-work\runs\gradcam\gradcam_samples.csv`
- `D:\luna16-work\runs\external_test\metrics_with_ci.csv`
- `D:\luna16-work\runs\stats\significance_tests.csv`
- `D:\luna16-work\runs\external_test\data_source_manifest.json`

- 主实验为 LUNA16 约 1:3 负采样后的候选分类，不应表述为官方全候选 FROC。
- 消融实验目前仅有 seed 0；跨 seed 稳定性应以主实验表为准。
- 当前 LIDC-IDRI 数据不足以开展正式二分类外部验证，固定测试结果不能改称 LIDC 外部结果。
