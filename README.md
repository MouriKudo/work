# PBIP-Lite 肺结节候选分类工程

本项目以 ResNet18 为特征主干，在 LUNA16/LIDC-IDRI 医学影像数据上实现候选级肺结节二分类，并集成 PBIP-Lite 原型学习、类别感知原型对比损失、Grad-CAM、Top-3 原型检索、六类图像退化评估、多随机种子统计和固定测试集 Bootstrap 置信区间。

> **评估边界**：当前系统对给定候选坐标做分类，不负责从整幅 CT 生成候选。预处理以约 1:3 保留正负候选，因此项目中的 FROC 是 **Sampled-Candidate FROC**，不能与官方 LUNA16 全候选检测 FROC 直接等同。

## 1. 已实现功能

- SimpleITK 读取 `.mhd/.raw`，世界坐标转体素坐标，抽取相邻三层 `3×64×64` patch；
- 肺窗裁剪（窗宽 1500 HU、窗位 -600 HU）和 `[0, 1]` 归一化；
- ResNet18、强增强 ResNet18、PBIP-Lite、PBIP + 原型对比损失训练与推理；
- LIDC-IDRI 去重可用性审计，不满足条件时自动回退到 LUNA16 固定测试集；
- Accuracy、Precision、Recall、F1、AUC、PR-AUC 与 CT 簇 Bootstrap 95% CI；
- Seed 0/1/2 均值±标准差、配对/双样本 t 检验、Wilcoxon 符号秩检验；
- 主实验、消融、鲁棒性、解释性和固定测试表格导出；
- 300 DPI PNG/SVG 论文图、Grad-CAM 四分类样本和 Top-3 病例检索；
- 真实小样本“加载 → 推理 → 评估 → 统计 → 绘图”一键冒烟测试。

## 2. 项目结构

```text
.
├── data/
│   ├── raw/                         # LUNA16 .mhd/.raw 与原始 CSV
│   └── processed/                   # patch、metadata 与外部样本
├── docs/
│   ├── methodology.md               # 方法、公式、伪代码、超参数
│   ├── background_and_related_work.md
│   ├── config_samples/              # 训练/推理 YAML 样例
│   └── SMOKE_TEST_LOG_TEMPLATE.md
├── paper_figs/task3/                # 8 组 PNG/SVG 论文图
├── runs/                            # checkpoint、预测、统计、表格与日志
├── src/                             # 预处理、训练、评估、统计与绘图代码
├── tests/                           # 单元和集成测试
├── smoke_test.sh                    # Bash/Git Bash 一键入口
├── smoke_test.py                    # 一键工程冒烟测试
├── requirements.txt                 # 经验证的准确依赖版本
└── FINAL_TECHNICAL_REPORT.md         # 完整技术报告
```

`paper_code/` 为原 PBIP 病理图像分割代码的只读方法参考；本项目实际入口均位于 `src/` 或仓库根目录。

## 3. 环境配置

已验证环境为 Python 3.10、PyTorch 2.11.0、CUDA 13.0、Windows 10/11 和 NVIDIA RTX 4060 Laptop GPU。依赖版本全部固定在 `requirements.txt`。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` 默认使用 PyTorch CUDA 13.0 wheel。CPU 或其他 CUDA 版本应先按目标平台安装匹配的 `torch/torchvision`，再安装其余依赖，不要混用不兼容的 CUDA wheel。

生成当前环境快照：

```powershell
.\.venv\Scripts\python.exe src\env_snapshot.py
```

快照保存到 `runs/environment/`。

## 4. 数据准备

### 4.1 LUNA16 目录

将 `.mhd/.raw` 放在 `data/raw/subset*/` 下，并准备 LUNA16 的 `annotations.csv` 与 `candidates.csv`。`.mhd` 头文件中的相对 `ElementDataFile` 必须能定位对应 `.raw`。

### 4.2 预处理与固定划分

```powershell
.\.venv\Scripts\python.exe src\data_manifest.py
.\.venv\Scripts\python.exe src\coordinate_utils.py
.\.venv\Scripts\python.exe src\preprocess.py --neg_ratio 3 --neg_strategy random --visualize
.\.venv\Scripts\python.exe src\split_data.py --visualize
```

默认保留全部正例，并按 1:3 抽样负例；可将 `--neg_strategy` 改为 `hard_negative`。现有划分按 CT/`seriesuid` 隔离：subset 0–7 训练、8 验证、9 固定测试，防止同一 CT 跨集合泄漏。

当前处理版本共 4512 个候选、863 个 CT：1128 个正例和 3384 个负例。固定测试集含 497 个候选、87 个 CT，其中正例 98、负例 399。

### 4.3 LIDC-IDRI 外部数据规则

外部评估只有在同时满足以下条件时才使用 LIDC-IDRI：

1. 已完成按 SeriesInstanceUID/SOPInstanceUID 和修订优先级去重；
2. 与 LUNA16 训练/验证/测试 CT 无重叠；
3. 同时存在正、负候选；
4. `.mhd`/patch/metadata 完整可读。

不满足时，`src/evaluate_external.py --data-source auto` 会记录原因并回退到 LUNA16 固定测试集，绝不把不完整 LIDC 样本伪装成外部性能结果。

## 5. 训练

### 5.1 单模型

```powershell
# 强增强 ResNet18
.\.venv\Scripts\python.exe src\train.py --epochs 20 --augment strong --seed 0

# 与基线权重匹配的原型库
.\.venv\Scripts\python.exe src\prototype_bank.py `
  --checkpoint runs\<strong-run>\best_model.pth `
  --output_dir runs\<prototype-bank>

# PBIP-Lite：beta=0
.\.venv\Scripts\python.exe src\pbip_train.py --epochs 20 --beta 0 `
  --prototype_bank runs\<prototype-bank>\prototype_bank.pkl `
  --init_checkpoint runs\<strong-run>\best_model.pth

# PBIP + 类别感知原型对比损失
.\.venv\Scripts\python.exe src\pbip_train.py --epochs 20 --beta 0.05 `
  --prototype_bank runs\<prototype-bank>\prototype_bank.pkl `
  --init_checkpoint runs\<strong-run>\best_model.pth
```

### 5.2 四方法 × 三随机种子

```powershell
.\.venv\Scripts\python.exe src\run_experiments.py `
  --output_root runs\experiments_v2 `
  --seeds 0,1,2 `
  --epochs 20 `
  --contrast_beta 0.05 `
  --beta_sweep 0.05,0.1
```

已有 `results.json` 的组合会被跳过，可断点续跑。配置样例见 `docs/config_samples/train.yaml`。

## 6. 推理教程

推理时先在验证集选择 F1 阈值，再锁定该阈值评估测试集。PBIP 模型必须使用与 checkpoint 对应的原型库。

```powershell
.\.venv\Scripts\python.exe src\metrics.py `
  --model pbip `
  --checkpoint runs\experiments_v2\seed_1\pbip_contrast\best_model.pth `
  --prototype_bank runs\experiments_v2\seed_1\prototype_bank\prototype_bank.pkl `
  --metadata data\processed\metadata.csv `
  --patches data\processed\patches `
  --output_dir runs\inference_example
```

输出包括验证阈值、测试预测 CSV、ROC、Sampled-Candidate FROC、固定 FP/scan 操作点和指标 JSON/CSV。单病例输入仍需先按 `src/preprocess.py` 的同一肺窗与三层 patch 规则预处理，不能把原始 HU 数组直接送入模型。

## 7. 固定测试与统计分析

```powershell
# 自动检查 LIDC-IDRI；不满足独立外部评估条件时回退 LUNA16 固定测试
.\.venv\Scripts\python.exe src\evaluate_external.py `
  --data-source auto `
  --methods resnet18_augmented,pbip_lite `
  --seeds 0,1,2 `
  --bootstrap-iterations 1000

# Bootstrap、Seed 统计、t 检验和 Wilcoxon 检验
.\.venv\Scripts\python.exe src\stats_test.py `
  --predictions runs\external_test\predictions.csv `
  --results runs\external_test\metrics_with_ci.csv `
  --method-a resnet18_augmented `
  --method-b pbip_lite `
  --seeds 0,1,2 `
  --bootstrap-iterations 1000
```

输出分别位于 `runs/external_test/` 和 `runs/stats/`。Bootstrap 以 CT/`seriesuid` 为重采样单位，避免把同一 CT 的多个候选错误视为完全独立样本。

## 8. 图表与结果表

所有图表和表格只读取实际日志/CSV，不内置虚构实验值。

```powershell
.\.venv\Scripts\python.exe src\make_figures.py --formats png,svg --dpi 300
.\.venv\Scripts\python.exe src\export_tables.py
```

- 图：`paper_figs/task3/`，含流程、结构、主指标、K 消融、AUC/F1 退化曲线、Grad-CAM、Top-3 检索；
- 表：`runs/summary_v3/tables/`；
- 中文结果分析：`runs/summary_v3/results_analysis.md`；
- 图表数据来源清单：`paper_figs/task3/figure_manifest.csv`。

## 9. 一键冒烟测试

默认命令使用真实 LUNA16 `.mhd`、处理后 patch、Seed 1 PBIP checkpoint 与对应原型库。Linux、macOS 或 Git Bash 执行：

```bash
bash smoke_test.sh
```

Windows PowerShell 也可直接调用 Python 入口：

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

该命令按顺序验证：

1. SimpleITK 读取真实 `.mhd`，并从验证/测试集各选 32 个类别平衡候选；
2. 加载 PBIP 模型与原型库，在验证小样本选择阈值并完成测试推理；
3. 计算分类指标；
4. 执行 200 次 CT 簇 Bootstrap 95% CI；
5. 生成 300 DPI 诊断仪表盘及结构化日志。

默认产物位于 `runs/smoke_test/latest/`：

```text
smoke_test.log          人可读运行日志
summary.json            环境、输入、阶段状态和总结果
step_status.csv         分阶段 PASS/FAIL 与耗时
selected_samples.csv    实际抽取的小样本
predictions.csv         候选级概率与预测
metrics.csv             点估计
bootstrap_ci.csv        95% CI
smoke_dashboard.png     ROC、PR、概率 ECDF、混淆矩阵
```

常用覆盖参数：

```powershell
.\.venv\Scripts\python.exe smoke_test.py `
  --device cpu `
  --sample-size 16 `
  --bootstrap-iterations 100 `
  --output-dir runs\smoke_test\cpu_check
```

冒烟指标只证明工程链路可运行，不能替代完整固定测试结果。人工归档模板见 `docs/SMOKE_TEST_LOG_TEMPLATE.md`。

## 10. 测试与质量检查

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m py_compile smoke_test.py src\evaluate_external.py src\stats_test.py src\make_figures.py src\export_tables.py
.\.venv\Scripts\python.exe -m pip check
```

测试覆盖坐标转换、数据集、原型分数与损失、阈值/FROC、退化、外部数据源回退、Bootstrap、统计检验、表格与图形数据校验。

## 11. 当前主要结果

结果均为固定测试集三随机种子均值±标准差，阈值只由对应验证集选择：

| 方法 | AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| ResNet18，无增强 | 0.98485 ± 0.00321 | 0.96364 ± 0.00368 | 0.89218 ± 0.01817 |
| ResNet18，强增强 | **0.99625 ± 0.00125** | **0.98814 ± 0.00244** | 0.92373 ± 0.01068 |
| PBIP-Lite，β=0 | 0.99500 ± 0.00008 | 0.98654 ± 0.00118 | **0.93569 ± 0.02123** |
| PBIP + 对比损失，β=0.05 | 0.99502 ± 0.00041 | 0.98661 ± 0.00149 | 0.93178 ± 0.01045 |

强增强 ResNet18 的平均 AUC/PR-AUC 最高；PBIP-Lite 的平均 F1 最高，但 F1 跨种子波动也更大。三种显著性检验均未发现强增强 ResNet18 与 PBIP-Lite 的五项固定测试指标达到 `p < 0.05`，因此不能声称总体性能存在统计显著差异。

## 12. 文档与可复现性

- 完整技术报告：[`FINAL_TECHNICAL_REPORT.md`](FINAL_TECHNICAL_REPORT.md)
- 方法说明：[`docs/methodology.md`](docs/methodology.md)
- 背景与相关工作：[`docs/background_and_related_work.md`](docs/background_and_related_work.md)
- 增量实验命令：[`INCREMENTAL_TASKS.md`](INCREMENTAL_TASKS.md)
- 结果分析：[`runs/summary_v3/results_analysis.md`](runs/summary_v3/results_analysis.md)

所有结论均应追溯到 `runs/` 下的实际 CSV/JSON/日志；若更换数据、划分、checkpoint 或原型库，应重新运行统计、绘图和导出脚本，避免报告与模型产物错配。
