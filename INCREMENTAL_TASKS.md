# 增量实验运行说明

本说明对应 K 值/组件消融、图像退化、鲁棒训练、Grad-CAM、病例检索和
LIDC-IDRI 外部验证。以下命令均在项目根目录执行。Windows 上建议显式使用项目
虚拟环境，避免系统 `python.exe` 应用别名干扰。

```powershell
$py = ".\.venv\Scripts\python.exe"
```

## 1. 四方法主结果与 K 值实验

统一重算四种方法 × seed 0/1/2 的 AUC、PR-AUC、验证阈值 F1：

```powershell
& $py src/summarize_results.py
```

输出：

- `runs/summary_v3/main_results.csv`：逐 seed 指标。
- `runs/summary_v3/main_results_summary.csv`：均值与标准差。
- `paper_figs/main_metrics_comparison.png`：指标柱状图。
- `runs/summary_v3/conclusion_template.md`：结论模板。

K=1/3/5 与其他消融共用下一节的批处理；训练完成后解析脚本生成
`runs/ablations/k_ablation_results.csv` 和 `paper_figs/k_ablation_curve.png`。

## 2. 组件、K 与 beta 消融

配置：`src/configs/ablation.yaml`。K=3、beta=0.05/0.1 和“无对比损失”会复用
已有正式实验；其余实验可断点续跑。

```powershell
& $py src/run_ablations.py --config src/configs/ablation.yaml
& $py src/parse_ablation_logs.py --config src/configs/ablation.yaml
```

可只运行指定项：

```powershell
& $py src/run_ablations.py --only k1 k5 beta_0.01 beta_0.2
```

输出：`runs/ablations/ablation_results.csv`、K/beta 子表、每组日志、配置/权重和
`paper_figs/k_ablation_curve.png`、`paper_figs/beta_ablation_curve.png`。

## 3. 六种 CT 退化

实现位于 `src/degradation.py`，强度参数在 `src/configs/degradation.yaml`。输入、
输出均保持 `C×H×W` 和 `[0,1]`，支持 NumPy/Tensor。

```powershell
& $py src/degradation.py --level 3 --samples 6
```

输出：`paper_figs/degradation_grid.png`（6×N 网格）。代码中可直接使用
`DegradationTransform` 或 `MixedDegradationTransform`。

## 4. 四方法退化鲁棒性

```powershell
& $py src/evaluate_robustness.py --config src/configs/robustness.yaml
```

每种方法从 seed 0/1/2 中只按验证 AUC 选择代表模型；阈值只在干净验证集选择
一次。输出 `runs/robustness/robustness_detailed.csv`、
`robustness_summary.csv`、选模记录，以及 AUC/F1 下降曲线。

## 5. Robust-PBIP

```powershell
& $py src/train_robust_pbip.py --config src/configs/robust_pbip.yaml
```

训练集混合干净样本、六种退化和既有强增强；验证集保持干净。输出最佳权重、
训练曲线和 `runs/robust_pbip/seed_0/clean_degraded_comparison.csv`。若已有权重只想
重评估，可增加 `--evaluate-only`。

## 6. Grad-CAM 四类样本

```powershell
& $py src/gradcam.py --methods resnet18_augmented pbip_full --samples-per-class 3
```

脚本在测试集自动筛选 TP/FP/FN/TN；目标为阳性 logit，热图覆盖三层输入的中心
层。输出筛选 CSV 和 `paper_figs/gradcam_*_four_categories.png`。

## 7. Top-3 相似病例检索

第一次运行会构建训练集和测试集特征索引：

```powershell
& $py src/retrieval.py --method pbip_full --query-index 0 --top-k 3
```

也可传入自定义 `.npy` patch：

```powershell
& $py src/retrieval.py --query-patch data/processed/patches/<patch>.npy
```

输出：`runs/retrieval/*_train_index.npz`、`*_test_index.npz`、检索 CSV 和
`paper_figs/retrieval_top3_example.png`。默认排除与查询相同的 seriesuid。

## 8. LIDC-IDRI 外部验证

下载官方 XML-only 包并生成校验清单：

```powershell
& $py src/download_lidc_sample.py --download-xml
```

先解析 XML，生成可用于筛选真实 sample series 的标注清单：

```powershell
& $py src/lidc_external.py `
  --lidc-root data/external/LIDC-IDRI `
  --xml-root data/external/LIDC-IDRI/xml `
  --dicom-root data/external/LIDC-IDRI/dicom
```

从 TCIA 官方 NBIA API 自动选择并下载体积最小的带标注、非 LUNA16 重叠 CT
series，然后执行真实 patch 提取：

```powershell
& $py src/download_lidc_sample.py --download-sample
& $py src/lidc_external.py `
  --lidc-root data/external/LIDC-IDRI `
  --xml-root data/external/LIDC-IDRI/xml `
  --dicom-root data/external/LIDC-IDRI/dicom `
  --extract
```

输出位于 `runs/external_validation/`，包括可行性报告、DICOM/XML 清单、UID 精确
重叠统计、下载 MD5 清单、非重叠病例和处理日志；patch 位于
`data/processed/lidc_external_patches/`。

当前真实 smoke series 为 `LIDC-IDRI-0957`，12 个 patch 全部使用 XML
`imageSOP_UID` 精确映射 DICOM 切片。完整 DICOM 约 124–133 GB，应在容量充足的
磁盘使用 TCIA Data Retriever 下载。LIDC XML 直接给出的是医师级结节标注；正式
外部 AUC/F1/FROC 还需要冻结阅片医师共识标签、负候选生成和匹配协议，不能只用
阳性 patch 宣称完成外部性能验证。

## 验证

新增模块可使用标准库测试运行，无需额外安装测试框架：

```powershell
& $py -m unittest discover -s tests -v
```

所有鲁棒性/FROC 结果仍属于当前约 1:3 负采样后的 sampled-candidate 评估，不应
表述为官方 LUNA16 全候选 FROC。
