# 消融实验结论模板

所有 F1 阈值仅在干净验证集选择；以下结果来自 seed 0，不能替代多 seed 主表。

- K 值实验中，测试 AUC 最佳为 K=3 （AUC=0.99542，PR-AUC=0.98827，F1=0.94301）。
- beta 实验中，测试 AUC 最佳为 beta=0.1 （AUC=0.99583，PR-AUC=0.98849，F1=0.93814）。
- no_prototype_logits：AUC=0.99468，PR-AUC=0.98738，F1=0.93750。
- no_contrastive_loss：AUC=0.99501，PR-AUC=0.98759，F1=0.91787。
- 需要结合验证曲线和跨 seed 结果讨论稳定性，不能仅按测试集最优值反向选择配置。
- 当前结论属于 sampled-candidate 分类，不是官方 LUNA16 全候选 FROC。
