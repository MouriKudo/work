# 冒烟测试运行日志模板

> 本模板用于人工归档 `smoke_test.py` 的一次运行。自动生成的原始日志位于
> `runs/smoke_test/<run>/smoke_test.log`，结构化结果位于 `summary.json`。

## 1. 基本信息

| 字段 | 记录 |
|---|---|
| 运行日期与时区 | `YYYY-MM-DD HH:MM:SS Asia/Shanghai` |
| 执行人/自动化任务 |  |
| Git 分支与 commit |  |
| 运行命令 | `python smoke_test.py ...` |
| 输出目录 |  |
| 最终状态 | `PASS / FAIL` |

## 2. 环境

| 字段 | 记录 |
|---|---|
| 操作系统 |  |
| Python |  |
| PyTorch / torchvision |  |
| CUDA runtime |  |
| GPU / CPU |  |
| SimpleITK |  |
| NumPy / pandas / SciPy |  |

## 3. 数据与模型

| 字段 | 记录 |
|---|---|
| metadata |  |
| patches 目录 |  |
| MHD 检查文件 |  |
| 验证/测试抽样数 |  |
| 测试 CT 数 |  |
| 测试类别计数 |  |
| 模型类型 |  |
| checkpoint |  |
| 原型库 |  |
| 随机种子 |  |

## 4. 阶段记录

| 阶段 | 状态 | 耗时（s） | 关键结果/错误摘要 |
|---|---|---:|---|
| 加载 |  |  |  |
| 推理 |  |  |  |
| 评估 |  |  |  |
| 统计 |  |  |  |
| 绘图 |  |  |  |

## 5. 冒烟指标

> 下列指标只用于工程链路验证，不得替代完整固定测试或外部验证结果。

| 指标 | 点估计 | 95% CI（若适用） |
|---|---:|---:|
| Accuracy |  |  |
| Precision |  |  |
| Recall |  |  |
| F1 |  |  |
| AUC |  |  |
| PR-AUC |  |  |

## 6. 产物核对

- [ ] `smoke_test.log`
- [ ] `summary.json`
- [ ] `selected_samples.csv`
- [ ] `predictions.csv`
- [ ] `metrics.csv`
- [ ] `bootstrap_ci.csv`
- [ ] `step_status.csv`
- [ ] `smoke_dashboard.png`

## 7. 异常、处置与结论

- 异常现象：
- 根因：
- 处置：
- 是否需要阻断发布：
- 复验结果：
- 审核人：

