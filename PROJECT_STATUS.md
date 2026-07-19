# 项目验收状态

本文件按原始任务清单记录可核验的完成状态。正式实验运行期间会持续更新。

## 数据与环境阶段

- [x] Git 仓库、Python/PyTorch/SimpleITK/scikit-learn/OpenCV/matplotlib 环境。
- [x] 原 PBIP 论文代码放入 `paper_code/`，仅作只读参考。
- [x] `data/raw`、`data/processed`、`src`、`runs`、`paper_figs` 目录。
- [x] LUNA16 subsets 0–9 全部解压，三个官方 CSV 存在。
- [x] 数据清单：`data/raw/data_manifest.csv`。
- [x] 环境报告、目录结构和截图：`runs/environment/`。
- [x] SimpleITK读取、世界坐标/体素坐标转换和64×64三层patch。
- [x] 坐标测试记录：`runs/data_audit/coordinate_tests.txt`。
- [x] 肺窗归一化、正例保留、随机/hard-negative负例采样。
- [x] 4512个processed patches和metadata。
- [x] metadata前100行：`runs/data_audit/metadata_head_100.csv`。
- [x] seriesuid级train/val/test划分，无患者/CT泄漏。
- [x] 类别、候选数、HU和划分统计及可视化：`paper_figs/`。

说明：原始subset ZIP已在解压后移除，因此 ZIP 归档本身的 MD5 无法从现有文件
恢复。当前清单为每个 subset 记录了解压目录总大小，以及由“相对路径、文件大小、
完整文件内容”共同计算的确定性树级 MD5；三个 CSV 保留文件级 MD5。哈希范围在
`md5_scope` 列中明确标注，避免与 ZIP MD5 混淆。

## 模型与实验阶段

- [x] Dataset、DataLoader、ResNet18训练入口。
- [x] 20 epoch ResNet18基线和增强实验。
- [x] loss/AUC/F1、数据泄漏、标签和候选坐标审计。
- [x] 轻量SimpleCNN代码和3 epoch流程验证。
- [x] 轻量SimpleCNN 20 epoch正式结果。
- [x] K=3、N=20的类别原型库和原型网格图。
- [x] 修正后的类别感知PBIP-Lite实现。
- [x] 使用标签且保持非负的原型对比损失。
- [x] 验证集最佳F1阈值、测试指标、ROC和sampled-candidate FROC/CPM。
- [x] β=0.05/0.1正式稳定性对比。
- [x] 四方法 × seed 0/1/2 共12组正式实验。
- [x] 最终 `results.csv`、`summary.csv` 和最佳模型统一评估。

## 当前正式矩阵

输出目录：`runs/experiments_v2/`

方法定义：

1. `resnet18_noaug`
2. `resnet18_strong`
3. `pbip_lite`（β=0）
4. `pbip_contrast`（β=0.05）

每个seed的PBIP使用该seed强增强ResNet18所构建的原型库，不复用其他seed的
特征空间。模型仅按验证AUC保存最佳权重，测试集不参与模型选择。

## 最终正式结果

| 方法 | 测试 AUC（均值±标准差） | 测试 F1（均值±标准差） |
| --- | ---: | ---: |
| ResNet18，无增强 | 0.98485 ± 0.00321 | 0.91456 ± 0.01668 |
| ResNet18，强增强 | **0.99625 ± 0.00125** | 0.93790 ± 0.00642 |
| PBIP-Lite，β=0 | 0.99500 ± 0.00008 | **0.94183 ± 0.02168** |
| PBIP + 对比损失，β=0.05 | 0.99502 ± 0.00041 | **0.94183 ± 0.02168** |

- 轻量 SimpleCNN（seed 42）：最佳验证 AUC 0.7825，测试 AUC 0.7802，
  测试 F1 0.2602。
- β 稳定性（seed 0）：β=0.05 的最佳验证 AUC 为 0.99140，β=0.1 为
  0.99135；前者验证指标略高且末轮总损失更低，因此主矩阵采用 β=0.05。
- 统一最佳模型按验证 AUC 选择为 seed 1 强增强 ResNet18。验证集最佳 F1 阈值为
  0.22029；测试 AUC 0.99494，在该阈值下 F1 0.91429、灵敏度 0.97959、
  特异度 0.95990。默认 0.5 阈值的测试 F1 为 0.93069。
- sampled-candidate CPM 为 0.99271。由于负候选按约 1:3 采样，该值不能当作
  官方 LUNA16 全候选 FROC/CPM。

最终产物位于：

- `runs/experiments_v2/results.csv` 与 `summary.csv`
- `runs/experiments_v2/beta_comparison.csv/png`
- `runs/experiments_v2/best_evaluation/`
- `runs/lightweight/simplecnn_strong_sd42/`

## 验证

```powershell
python -m unittest discover -s tests -v
```

当前共12个测试，覆盖数据无泄漏、patch范围、清单状态、原型分类、对比损失、
余弦聚类、阈值/FROC和实验矩阵定义。测试记录在
`runs/validation/unit_tests.txt`。
