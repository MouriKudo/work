# LUNA16 PBIP-Lite 肺结节候选分类

本项目研究如何把原型学习思想用于 LUNA16 肺结节候选分类。输入是 LUNA16
提供的候选坐标及其相邻三层 CT patch，输出是候选为真实肺结节的概率。

> 本项目是候选级二分类，不是从整幅 CT 直接生成候选的端到端检测器。
> 由于预处理阶段将负候选采样为约 1:3，项目中的 FROC 是
> **sampled-candidate FROC**，不能直接当作官方 LUNA16 全候选 FROC。

## 数据流程

1. 用 SimpleITK 读取 `.mhd/.raw` CT。
2. 将 `candidates.csv` 世界坐标转换为体素坐标。
3. 在候选中心提取相邻三层 `(3, 64, 64)` patch。
4. 使用肺窗（窗宽 1500、窗位 -600）裁剪并归一化到 `[0, 1]`。
5. 与 `annotations.csv` 在 5 mm 容差内匹配正例；保留全部正例，并随机或
   hard-negative 采样负例。
6. 按 seriesuid/CT 划分：subsets 0–7 训练、8 验证、9 测试。

当前处理数据包含 4512 个候选、863 个 CT，其中正例 1128、负例 3384。

## 方法

- `resnet18_noaug`：无增强 ResNet18。
- `resnet18_strong`：旋转、翻转、裁剪、强度扰动和模糊的 ResNet18。
- `pbip_lite`：正负类别原型余弦相似度差分融合，`beta=0`。
- `pbip_contrast`：PBIP-Lite 加类别感知原型对比损失。
- `simplecnn_strong`：轻量 SimpleCNN 检查模型复杂度影响。

`paper_code/` 是原始 PBIP 病理图像分割代码，仅作为只读方法参考；LUNA16
代码位于 `src/`，不直接修改或依赖原论文训练入口。

## 主要目录

```text
data/raw/                 LUNA16 CT、annotations.csv、candidates.csv
data/processed/           三层 patch 和 metadata.csv
paper_code/               原论文参考代码（只读）
paper_figs/               数据统计与样本图
runs/                     模型、日志、指标和评估输出
src/                      预处理、训练、原型库和评估代码
tests/                    快速单元测试
```

## 安装

Python 3.10 和 CUDA 13.0 环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

实际环境版本、目录结构和截图可重新生成：

```powershell
python src/env_snapshot.py
```

输出位于 `runs/environment/`。

## 数据准备

```powershell
python src/data_manifest.py
python src/coordinate_utils.py
python src/preprocess.py --neg_ratio 3 --neg_strategy random --visualize
python src/split_data.py --visualize
```

如需 hard negative，把 `--neg_strategy` 改为 `hard_negative`。重新预处理会改写
patch 和 metadata，运行前应保留现有实验所对应的数据版本。

现有 ZIP 已在解压后删除。若要复核当前解压数据的完整内容 MD5（不是 ZIP
归档 MD5），运行：

```powershell
python src/data_manifest.py --hash_extracted
```

## 单模型训练

三轮流程验证：

```powershell
python src/train.py --epochs 3 --augment strong --seed 0
```

20轮 ResNet18：

```powershell
python src/train.py --epochs 20 --augment strong --seed 0
```

轻量基线：

```powershell
python src/train_lightweight.py --epochs 20 --augment strong --seed 42
```

构建与基线权重匹配的原型库：

```powershell
python src/prototype_bank.py `
  --checkpoint runs/<strong-run>/best_model.pth `
  --output_dir runs/<prototype-bank>
```

PBIP-Lite 与对比损失：

```powershell
# 纯原型融合
python src/pbip_train.py --epochs 20 --beta 0 `
  --prototype_bank runs/<prototype-bank>/prototype_bank.pkl `
  --init_checkpoint runs/<strong-run>/best_model.pth

# 原型对比损失
python src/pbip_train.py --epochs 20 --beta 0.05 `
  --prototype_bank runs/<prototype-bank>/prototype_bank.pkl `
  --init_checkpoint runs/<strong-run>/best_model.pth
```

## 四方法 × 三seed实验

以下命令执行 seed 0、1、2 的完整12组矩阵，并额外完成 β=0.05/0.1
参考seed对照。每个seed会从自己的强增强模型构建原型库。命令支持断点续跑，
已有 `results.json` 的方法会自动跳过。

```powershell
python src/run_experiments.py `
  --output_root runs/experiments_v2 `
  --seeds 0,1,2 `
  --epochs 20 `
  --contrast_beta 0.05 `
  --beta_sweep 0.05,0.1
```

汇总输出：

- `runs/experiments_v2/results.csv`：12组原始结果。
- `runs/experiments_v2/summary.csv`：各方法均值和标准差。
- `runs/experiments_v2/beta_comparison.csv/png`：β稳定性对比。
- `runs/experiments_v2/best_evaluation/`：仅按验证 AUC 选择的最终模型评估。

## 正式实验结果

下表统一用干净验证集选择各 seed 的 F1 阈值，再锁定到测试集。它与训练脚本中
固定 0.5 阈值的历史 `test_metrics` 分开保留，不能混用。

| 方法 | 测试 AUC（均值±标准差） | 测试 PR-AUC（均值±标准差） | 测试 F1（均值±标准差） |
| --- | ---: | ---: | ---: |
| ResNet18，无增强 | 0.98485 ± 0.00321 | 0.96364 ± 0.00368 | 0.89218 ± 0.01817 |
| ResNet18，强增强 | **0.99625 ± 0.00125** | **0.98814 ± 0.00244** | 0.92373 ± 0.01068 |
| PBIP-Lite，β=0 | 0.99500 ± 0.00008 | 0.98654 ± 0.00118 | **0.93569 ± 0.02123** |
| PBIP + 对比损失，β=0.05 | 0.99502 ± 0.00041 | 0.98661 ± 0.00149 | 0.93178 ± 0.01045 |

强增强 ResNet18 的平均 AUC/PR-AUC 最好；PBIP-Lite 的平均 F1 略高，但跨 seed
F1 方差也更大。β=0.05 与 0.1 在 seed 0 的最佳验证 AUC 分别为 0.99140 和
0.99135，因此主方法保留 β=0.05。轻量 SimpleCNN 的测试 AUC 为 0.7802、
F1 为 0.2602，说明主结果并非仅由数据流程本身轻易获得。

## 阈值与FROC

阈值只在验证集选择，之后锁定并应用到测试集：

```powershell
python src/metrics.py `
  --model resnet18 `
  --checkpoint runs/<run>/best_model.pth `
  --output_dir runs/<evaluation>
```

输出包括最佳阈值、默认/调优阈值测试指标、候选预测、ROC、FROC、固定
FP/scan操作点及 sampled-candidate CPM。

最终统一评估仅按验证 AUC 选中 seed 1 强增强 ResNet18。验证集最佳 F1 阈值
为 0.22029；测试 AUC 为 0.99494，调优阈值下 F1 为 0.91429、灵敏度
0.97959、特异度 0.95990。默认 0.5 阈值下测试 F1 为 0.93069。阈值在验证集
上优化并不保证测试 F1 必然高于默认阈值，因此两个结果都保留。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖类别感知原型分数、原型对比损失、阈值/FROC、余弦聚类和批量实验矩阵。

## 增量实验：消融、退化鲁棒性与可解释检索

新增任务的完整运行顺序、配置入口和产物位置见
[`INCREMENTAL_TASKS.md`](INCREMENTAL_TASKS.md)。所有新增实验都由 argparse 或
`src/configs/*.yaml` 驱动，表格写入 `runs/`，论文图写入 `paper_figs/`。

增量实验已完成消融、六类五档退化评估、Robust-PBIP、Grad-CAM 和 Top-3
检索。Robust-PBIP 提升了 Level 3 退化宏平均 AUC，但固定验证阈值下 F1 低于
普通 PBIP，不能表述为所有指标全面提升。

LIDC-IDRI 可行性流水线已使用真实数据验证：官方 XML-only 包共解析 1,319 个
XML，按 SeriesInstanceUID 和 correction 优先规则得到 1,294 个唯一 XML series、
20,096 条医师级结节标注；真实非 LUNA16 重叠病例 `LIDC-IDRI-0957` 成功提取
12 个 3×64×64 patch，全部按 SOP UID 精确对齐且无失败。下载校验记录见
`runs/external_validation/source_manifest.csv`。这证明数据处理链路可用，但不等同于
完整 LIDC-IDRI 外部性能验证；正式 AUC/F1/FROC 仍需完整 DICOM、跨医师共识和
负候选协议。

数据来源：[TCIA LIDC-IDRI collection](https://www.cancerimagingarchive.net/collection/lidc-idri/)
（CC BY 3.0）；DICOM sample 通过官方
[NBIA Search REST API](https://wiki.cancerimagingarchive.net/display/Public/NBIA+Search+REST+API+Guide)
获取。
