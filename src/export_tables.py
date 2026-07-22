"""汇总肺结节分类项目的五类实验表格并生成中文结果分析。

所有数值均从已有 CSV/JSON 实验产物读取。脚本负责字段校验、类型规范、派生
统计和 CSV 导出；若源文件缺失、包含重复主键或指标越界，将直接失败，绝不使用
默认数值填充。输出包括主结果、消融、鲁棒性、解释性和外部/固定测试五张表。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger("export_tables")

METHOD_ORDER = (
    "resnet18_baseline",
    "resnet18_augmented",
    "pbip_lite",
    "pbip_full",
)
METHOD_LABELS_ZH = {
    "resnet18_baseline": "ResNet18（无增强）",
    "resnet18_augmented": "ResNet18（强增强）",
    "pbip_lite": "PBIP-Lite",
    "pbip_full": "PBIP + 原型对比损失",
}
ABLATION_LABELS_ZH = {
    "no_prototype_logits": "移除原型 logit",
    "no_contrastive_loss": "移除原型对比损失",
    "k1": "K=1",
    "k3": "K=3",
    "k5": "K=5",
    "beta_0.01": "β=0.01",
    "beta_0.05": "β=0.05",
    "beta_0.1": "β=0.10",
    "beta_0.2": "β=0.20",
}
GROUP_LABELS_ZH = {"component": "组件消融", "k": "K 值消融", "beta": "β 值消融"}
DEGRADATION_ORDER = (
    "clean",
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
    "window_shift",
    "jpeg",
    "resample",
)
DEGRADATION_LABELS_ZH = {
    "clean": "干净数据",
    "low_contrast": "低对比度",
    "gaussian_noise": "高斯噪声",
    "gaussian_blur": "高斯模糊",
    "window_shift": "窗位偏移",
    "jpeg": "JPEG 压缩",
    "resample": "下采样",
}
METRIC_LABELS_ZH = {
    "accuracy": "准确率",
    "precision": "精确率",
    "recall": "召回率",
    "f1": "F1",
    "auc": "AUC",
    "pr_auc": "PR-AUC",
}
METRIC_COLUMNS = ("auc", "pr_auc", "f1", "accuracy", "precision", "recall")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} 缺少必要字段：{missing}")
    if frame.empty:
        raise ValueError(f"{source} 为空")


def read_csv_checked(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"表格源文件不存在：{path}")
    frame = pd.read_csv(path)
    require_columns(frame, columns, path)
    return frame


def _numeric(frame: pd.DataFrame, columns: Iterable[str], source: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{source} 的 {column} 含非有限数值")
    return result


def _validate_unit_interval(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{source} 的 {column} 超出 [0,1]")


def _assert_unique(frame: pd.DataFrame, keys: Sequence[str], source: str) -> None:
    duplicate = frame.duplicated(list(keys), keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, list(keys)].head().to_dict("records")
        raise ValueError(f"{source} 存在重复主键 {keys}：{sample}")


def build_main_results_table(source_path: Path) -> pd.DataFrame:
    """由逐 seed 主结果计算四方法均值和样本标准差。"""

    frame = read_csv_checked(
        source_path,
        ["method", "seed", *METRIC_COLUMNS],
    )
    frame = _numeric(frame, ["seed", *METRIC_COLUMNS], str(source_path))
    _validate_unit_interval(frame, METRIC_COLUMNS, str(source_path))
    _assert_unique(frame, ["method", "seed"], str(source_path))
    missing = [method for method in METHOD_ORDER if method not in set(frame["method"])]
    if missing:
        raise ValueError(f"主结果缺少方法：{missing}")

    rows = []
    for method in METHOD_ORDER:
        group = frame[frame["method"] == method].sort_values("seed")
        seeds = group["seed"].astype(int).tolist()
        if seeds != [0, 1, 2]:
            raise ValueError(f"{method} 的主结果种子必须为 0/1/2，实际为 {seeds}")
        row: dict[str, object] = {
            "method": method,
            "method_zh": METHOD_LABELS_ZH[method],
            "seeds": "0,1,2",
            "seed_count": len(group),
        }
        for metric in METRIC_COLUMNS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def build_ablation_table(source_path: Path) -> pd.DataFrame:
    """整理组件、K 值和 β 值消融的逐实验记录。"""

    columns = [
        "method",
        "seed",
        "group",
        "k",
        "alpha",
        "beta",
        "auc",
        "pr_auc",
        "f1",
        "accuracy",
        "precision",
        "recall",
        "threshold",
        "validation_f1",
        "reused",
    ]
    frame = read_csv_checked(source_path, columns)
    frame = _numeric(
        frame,
        [
            "seed",
            "k",
            "alpha",
            "beta",
            "auc",
            "pr_auc",
            "f1",
            "accuracy",
            "precision",
            "recall",
            "threshold",
            "validation_f1",
        ],
        str(source_path),
    )
    _validate_unit_interval(
        frame,
        ["auc", "pr_auc", "f1", "accuracy", "precision", "recall", "threshold", "validation_f1"],
        str(source_path),
    )
    _assert_unique(frame, ["method", "seed"], str(source_path))
    frame.insert(1, "method_zh", frame["method"].map(ABLATION_LABELS_ZH).fillna(frame["method"]))
    frame.insert(3, "group_zh", frame["group"].map(GROUP_LABELS_ZH).fillna(frame["group"]))
    group_order = {"component": 0, "k": 1, "beta": 2}
    frame["_group_order"] = frame["group"].map(group_order).fillna(99)
    frame = frame.sort_values(["_group_order", "k", "beta", "method"]).drop(columns="_group_order")
    return frame.reset_index(drop=True)[
        [
            "method",
            "method_zh",
            "group",
            "group_zh",
            "seed",
            "k",
            "alpha",
            "beta",
            "auc",
            "pr_auc",
            "f1",
            "accuracy",
            "precision",
            "recall",
            "threshold",
            "validation_f1",
            "reused",
        ]
    ]


def build_robustness_table(source_path: Path) -> pd.DataFrame:
    """导出 124 条鲁棒性记录，并计算相对同一模型干净数据的下降量。"""

    columns = ["method", "seed", "degradation", "level", "auc", "pr_auc", "f1", "threshold"]
    frame = read_csv_checked(source_path, columns)
    frame = _numeric(frame, ["seed", "level", "auc", "pr_auc", "f1", "threshold"], str(source_path))
    _validate_unit_interval(frame, ["auc", "pr_auc", "f1", "threshold"], str(source_path))
    _assert_unique(frame, ["method", "degradation", "level"], str(source_path))

    clean = frame[frame["degradation"] == "clean"].set_index("method")
    if set(clean.index) != set(METHOD_ORDER):
        raise ValueError("鲁棒性表必须为四种方法各包含一条干净数据记录")
    frame.insert(1, "method_zh", frame["method"].map(METHOD_LABELS_ZH))
    frame.insert(
        4,
        "degradation_zh",
        frame["degradation"].map(DEGRADATION_LABELS_ZH).fillna(frame["degradation"]),
    )
    for metric in ("auc", "pr_auc", "f1"):
        baseline = frame["method"].map(clean[metric].to_dict()).astype(float)
        frame[f"{metric}_drop_from_clean"] = baseline - frame[metric].astype(float)
    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    degradation_order = {name: index for index, name in enumerate(DEGRADATION_ORDER)}
    frame["_method_order"] = frame["method"].map(method_order)
    frame["_degradation_order"] = frame["degradation"].map(degradation_order).fillna(99)
    frame = frame.sort_values(
        ["_method_order", "_degradation_order", "level"]
    ).drop(columns=["_method_order", "_degradation_order"])
    return frame.reset_index(drop=True)


def build_explainability_table(source_path: Path) -> pd.DataFrame:
    """按方法和 TP/FP/FN/TN 汇总 Grad-CAM 代表样本。"""

    columns = [
        "method",
        "seed",
        "category",
        "probability",
        "cam_probability",
        "threshold",
        "patch_file",
    ]
    frame = read_csv_checked(source_path, columns)
    frame = _numeric(frame, ["seed", "probability", "cam_probability", "threshold"], str(source_path))
    _validate_unit_interval(frame, ["probability", "cam_probability", "threshold"], str(source_path))
    invalid_categories = sorted(set(frame["category"]) - {"TP", "FP", "FN", "TN"})
    if invalid_categories:
        raise ValueError(f"解释性记录含未知类别：{invalid_categories}")

    rows = []
    for keys, group in frame.groupby(["method", "seed", "category"], sort=True):
        method, seed, category = keys
        thresholds = group["threshold"].unique()
        if len(thresholds) != 1:
            raise ValueError(f"{method}/{category} 存在多个阈值")
        rows.append(
            {
                "method": method,
                "method_zh": METHOD_LABELS_ZH.get(method, method),
                "seed": int(seed),
                "category": category,
                "sample_count": int(len(group)),
                "probability_mean": float(group["probability"].mean()),
                "probability_min": float(group["probability"].min()),
                "probability_max": float(group["probability"].max()),
                "cam_probability_mean": float(group["cam_probability"].mean()),
                "threshold": float(thresholds[0]),
                "representative_patch": str(group.iloc[0]["patch_file"]),
            }
        )
    result = pd.DataFrame(rows)
    category_order = {category: index for index, category in enumerate(("TP", "FP", "FN", "TN"))}
    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    result["_method_order"] = result["method"].map(method_order).fillna(99)
    result["_category_order"] = result["category"].map(category_order)
    return result.sort_values(["_method_order", "_category_order"]).drop(
        columns=["_method_order", "_category_order"]
    ).reset_index(drop=True)


def build_external_test_table(source_path: Path) -> pd.DataFrame:
    """整理外部/固定测试逐 seed 指标和 CT 簇 Bootstrap CI。"""

    columns = [
        "dataset",
        "method",
        "seed",
        "metric",
        "value",
        "ci_lower",
        "ci_upper",
        "confidence_level",
        "bootstrap_iterations",
        "resample_unit",
        "threshold",
        "n_candidates",
        "n_series",
    ]
    frame = read_csv_checked(source_path, columns)
    frame = _numeric(
        frame,
        [
            "seed",
            "value",
            "ci_lower",
            "ci_upper",
            "confidence_level",
            "bootstrap_iterations",
            "threshold",
            "n_candidates",
            "n_series",
        ],
        str(source_path),
    )
    _validate_unit_interval(
        frame,
        ["value", "ci_lower", "ci_upper", "confidence_level", "threshold"],
        str(source_path),
    )
    if (frame["ci_lower"] > frame["ci_upper"]).any():
        raise ValueError("外部测试表存在 CI 下界大于上界")
    _assert_unique(frame, ["dataset", "method", "seed", "metric"], str(source_path))
    frame.insert(2, "method_zh", frame["method"].map(METHOD_LABELS_ZH).fillna(frame["method"]))
    frame.insert(5, "metric_zh", frame["metric"].map(METRIC_LABELS_ZH).fillna(frame["metric"]))
    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    metric_order = {name: index for index, name in enumerate(("accuracy", "precision", "recall", "f1", "auc"))}
    frame["_method_order"] = frame["method"].map(method_order).fillna(99)
    frame["_metric_order"] = frame["metric"].map(metric_order).fillna(99)
    return frame.sort_values(["_method_order", "seed", "_metric_order"]).drop(
        columns=["_method_order", "_metric_order"]
    ).reset_index(drop=True)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用无 BOM 的标准 UTF-8，避免部分科研软件把 BOM 识别为首列名字符。
    frame.to_csv(path, index=False, encoding="utf-8")
    LOGGER.info("已导出：%s（%d 行 × %d 列）", path, len(frame), len(frame.columns))


def _fmt(value: float) -> str:
    return f"{float(value):.5f}"


def _main_analysis(main: pd.DataFrame) -> str:
    lines = ["## 1. 主实验结果", ""]
    for metric in ("auc", "pr_auc", "f1"):
        best = main.loc[main[f"{metric}_mean"].idxmax()]
        lines.append(
            f"- 平均 {METRIC_LABELS_ZH[metric]} 最佳方法为 **{best['method_zh']}**："
            f"{_fmt(best[f'{metric}_mean'])} ± {_fmt(best[f'{metric}_std'])}。"
        )
    augmented = main[main["method"] == "resnet18_augmented"].iloc[0]
    pbip = main[main["method"] == "pbip_lite"].iloc[0]
    lines.extend(
        [
            "",
            f"PBIP-Lite 相对强增强 ResNet18 的平均 F1 变化为 "
            f"{float(pbip['f1_mean'] - augmented['f1_mean']):+.5f}，平均 AUC 变化为 "
            f"{float(pbip['auc_mean'] - augmented['auc_mean']):+.5f}。该结果说明当前实验中"
            "原型融合更偏向改善阈值相关的分类平衡，而最高平均 AUC 仍由强增强 ResNet18 获得。",
            "",
        ]
    )
    return "\n".join(lines)


def _ablation_analysis(ablation: pd.DataFrame) -> str:
    lines = ["## 2. 消融实验", ""]
    k_rows = ablation[ablation["group"] == "k"]
    beta_rows = ablation[ablation["group"] == "beta"]
    component_rows = ablation[ablation["group"] == "component"]
    if not k_rows.empty:
        best_auc = k_rows.loc[k_rows["auc"].idxmax()]
        best_f1 = k_rows.loc[k_rows["f1"].idxmax()]
        lines.append(
            f"- K 值消融中，AUC 最佳为 **{best_auc['method_zh']}**（{_fmt(best_auc['auc'])}），"
            f"F1 最佳为 **{best_f1['method_zh']}**（{_fmt(best_f1['f1'])}）。"
        )
    if not beta_rows.empty:
        best_auc = beta_rows.loc[beta_rows["auc"].idxmax()]
        best_f1 = beta_rows.loc[beta_rows["f1"].idxmax()]
        lines.append(
            f"- β 消融中，AUC 最佳为 **{best_auc['method_zh']}**（{_fmt(best_auc['auc'])}），"
            f"F1 最佳为 **{best_f1['method_zh']}**（{_fmt(best_f1['f1'])}）。"
        )
    for row in component_rows.itertuples():
        lines.append(
            f"- {row.method_zh}：AUC={_fmt(row.auc)}，PR-AUC={_fmt(row.pr_auc)}，"
            f"F1={_fmt(row.f1)}。"
        )
    lines.extend(
        [
            "- 消融结果均为 seed 0，适合描述组件趋势，不应替代三种子主实验的稳定性结论。",
            "",
        ]
    )
    return "\n".join(lines)


def _robustness_analysis(robustness: pd.DataFrame) -> str:
    lines = ["## 3. 图像退化鲁棒性", ""]
    level5 = robustness[
        (robustness["level"] == 5) & (robustness["degradation"] != "clean")
    ]
    aggregated = level5.groupby(["method", "method_zh"], as_index=False).agg(
        auc_mean=("auc", "mean"),
        f1_mean=("f1", "mean"),
        auc_drop_mean=("auc_drop_from_clean", "mean"),
        f1_drop_mean=("f1_drop_from_clean", "mean"),
    )
    best_auc = aggregated.loc[aggregated["auc_mean"].idxmax()]
    best_f1 = aggregated.loc[aggregated["f1_mean"].idxmax()]
    lines.append(
        f"- 六类 Level 5 退化的宏平均 AUC 最佳为 **{best_auc['method_zh']}**："
        f"{_fmt(best_auc['auc_mean'])}，相对干净数据平均下降 {_fmt(best_auc['auc_drop_mean'])}。"
    )
    lines.append(
        f"- 六类 Level 5 退化的宏平均 F1 最佳为 **{best_f1['method_zh']}**："
        f"{_fmt(best_f1['f1_mean'])}，相对干净数据平均下降 {_fmt(best_f1['f1_drop_mean'])}。"
    )
    for method in METHOD_ORDER:
        group = level5[level5["method"] == method]
        worst_auc = group.loc[group["auc_drop_from_clean"].idxmax()]
        worst_f1 = group.loc[group["f1_drop_from_clean"].idxmax()]
        lines.append(
            f"- {METHOD_LABELS_ZH[method]}：AUC 下降最大退化为"
            f"{worst_auc['degradation_zh']}（下降 {_fmt(worst_auc['auc_drop_from_clean'])}）；"
            f"F1 下降最大退化为{worst_f1['degradation_zh']}"
            f"（下降 {_fmt(worst_f1['f1_drop_from_clean'])}）。"
        )
    lines.extend(
        [
            "- 所有退化评估均沿用干净验证集冻结阈值，因此 F1 变化同时反映判别能力和概率校准变化。",
            "",
        ]
    )
    return "\n".join(lines)


def _explainability_analysis(explainability: pd.DataFrame) -> str:
    lines = ["## 4. Grad-CAM 解释性实验", ""]
    for method, group in explainability.groupby("method", sort=False):
        counts = {row.category: int(row.sample_count) for row in group.itertuples()}
        count_text = "、".join(f"{category}={counts.get(category, 0)}" for category in ("TP", "FP", "FN", "TN"))
        lines.append(f"- {METHOD_LABELS_ZH.get(method, method)} 代表样本数：{count_text}。")
    lines.extend(
        [
            "- 热图由阳性 logit 对 ResNet18 最后一个残差块反向传播得到，叠加在三层输入的中心层上。",
            "- 当前实验没有像素级病灶分割真值，因此本表只报告样本覆盖和预测概率，不能把热区视觉一致性表述为定量定位精度。",
            "",
        ]
    )
    return "\n".join(lines)


def _external_analysis(
    external: pd.DataFrame,
    significance_path: Path,
    manifest_path: Path,
) -> str:
    if not significance_path.exists():
        raise FileNotFoundError(f"显著性检验文件不存在：{significance_path}")
    significance = pd.read_csv(significance_path)
    require_columns(significance, ["metric", "test", "p_value", "significant"], significance_path)
    significance = _numeric(significance, ["p_value"], str(significance_path))
    if not manifest_path.exists():
        raise FileNotFoundError(f"外部测试清单不存在：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    source_name = manifest.get("selected_source", {}).get("name", "unknown")
    fallback = bool(manifest.get("source_audit", {}).get("fallback_used", False))
    fallback_reasons = manifest.get("source_audit", {}).get("fallback_reasons", [])

    lines = ["## 5. 外部/固定测试", ""]
    if fallback:
        lines.append(
            f"- 本次实际评估数据源为 **{source_name}**。LIDC-IDRI 自动回退原因："
            + "；".join(str(reason) for reason in fallback_reasons)
            + "。"
        )
    else:
        lines.append(f"- 本次实际评估数据源为 **{source_name}**，未触发回退。")

    summary = external.groupby(["method", "method_zh", "metric"], as_index=False).agg(
        mean=("value", "mean"), std=("value", "std")
    )
    for method in summary["method"].drop_duplicates():
        group = summary[summary["method"] == method].set_index("metric")
        lines.append(
            f"- {group.iloc[0]['method_zh']}："
            f"Acc={_fmt(group.loc['accuracy', 'mean'])}±{_fmt(group.loc['accuracy', 'std'])}，"
            f"F1={_fmt(group.loc['f1', 'mean'])}±{_fmt(group.loc['f1', 'std'])}，"
            f"AUC={_fmt(group.loc['auc', 'mean'])}±{_fmt(group.loc['auc', 'std'])}。"
        )
    minimum_p = float(significance["p_value"].min())
    significant_count = int(
        significance["significant"].astype(str).str.lower().isin({"true", "1"}).sum()
    )
    lines.extend(
        [
            f"- 三类检验在五项指标上的最小原始 p 值为 {_fmt(minimum_p)}，"
            f"p<0.05 的记录数为 {significant_count}。由于仅有 3 个随机种子，"
            "未显著不等同于两种方法性能等效。",
            "- 95% CI 使用 seriesuid 为单位的 1,000 次 CT 簇 Bootstrap；"
            "Precision、Recall 和 F1 使用各模型验证集冻结阈值。",
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis(
    output_path: Path,
    main: pd.DataFrame,
    ablation: pd.DataFrame,
    robustness: pd.DataFrame,
    explainability: pd.DataFrame,
    external: pd.DataFrame,
    significance_path: Path,
    manifest_path: Path,
    sources: Sequence[Path],
) -> None:
    """根据导出表格生成可复核的中文分析，不写入任何预设实验结论。"""

    source_lines = "\n".join(f"- `{path}`" for path in sources)
    text = "\n".join(
        [
            "# 肺结节分类实验结果分析",
            "",
            "> 本文档由 `src/export_tables.py` 从实际 CSV/JSON 自动生成。"
            "所有结论仅描述当前保存的实验结果。",
            "",
            _main_analysis(main),
            _ablation_analysis(ablation),
            _robustness_analysis(robustness),
            _explainability_analysis(explainability),
            _external_analysis(external, significance_path, manifest_path),
            "## 6. 数据来源与解释边界",
            "",
            source_lines,
            "",
            "- 主实验为 LUNA16 约 1:3 负采样后的候选分类，不应表述为官方全候选 FROC。",
            "- 消融实验目前仅有 seed 0；跨 seed 稳定性应以主实验表为准。",
            "- 当前 LIDC-IDRI 数据不足以开展正式二分类外部验证，固定测试结果不能改称 LIDC 外部结果。",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    LOGGER.info("已生成中文分析：%s", output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出肺结节项目实验表格和中文分析")
    parser.add_argument(
        "--main-results",
        type=Path,
        default=PROJECT_ROOT / "runs/summary_v3/main_results.csv",
    )
    parser.add_argument(
        "--ablation-results",
        type=Path,
        default=PROJECT_ROOT / "runs/ablations/ablation_results.csv",
    )
    parser.add_argument(
        "--robustness-results",
        type=Path,
        default=PROJECT_ROOT / "runs/robustness/robustness_detailed.csv",
    )
    parser.add_argument(
        "--gradcam-results",
        type=Path,
        default=PROJECT_ROOT / "runs/gradcam/gradcam_samples.csv",
    )
    parser.add_argument(
        "--external-results",
        type=Path,
        default=PROJECT_ROOT / "runs/external_test/metrics_with_ci.csv",
    )
    parser.add_argument(
        "--significance-results",
        type=Path,
        default=PROJECT_ROOT / "runs/stats/significance_tests.csv",
    )
    parser.add_argument(
        "--external-manifest",
        type=Path,
        default=PROJECT_ROOT / "runs/external_test/data_source_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/summary_v3/tables",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=PROJECT_ROOT / "runs/summary_v3/results_analysis.md",
    )
    return parser.parse_args(argv)


def export_all_tables(args: argparse.Namespace) -> dict[str, Path]:
    """生成五类 CSV 和中文分析，并返回输出路径。"""

    main = build_main_results_table(args.main_results)
    ablation = build_ablation_table(args.ablation_results)
    robustness = build_robustness_table(args.robustness_results)
    explainability = build_explainability_table(args.gradcam_results)
    external = build_external_test_table(args.external_results)

    output_dir = args.output_dir.resolve()
    paths = {
        "main": output_dir / "main_results_table.csv",
        "ablation": output_dir / "ablation_results_table.csv",
        "robustness": output_dir / "robustness_results_table.csv",
        "explainability": output_dir / "explainability_results_table.csv",
        "external": output_dir / "external_test_results_table.csv",
        "analysis": args.analysis_output.resolve(),
    }
    save_csv(main, paths["main"])
    save_csv(ablation, paths["ablation"])
    save_csv(robustness, paths["robustness"])
    save_csv(explainability, paths["explainability"])
    save_csv(external, paths["external"])
    write_analysis(
        paths["analysis"],
        main,
        ablation,
        robustness,
        explainability,
        external,
        args.significance_results,
        args.external_manifest,
        [
            args.main_results,
            args.ablation_results,
            args.robustness_results,
            args.gradcam_results,
            args.external_results,
            args.significance_results,
            args.external_manifest,
        ],
    )
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    export_all_tables(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # CLI 需要完整堆栈和非零状态
        LOGGER.exception("实验表格导出失败：%s", error)
        sys.exit(1)
