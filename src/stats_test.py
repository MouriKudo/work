"""肺结节分类实验的统计分析工具。

本脚本提供三类可复用能力：

1. 按 CT（``seriesuid``）进行簇 Bootstrap，并计算分类指标置信区间；
2. 汇总多随机种子实验的均值与样本标准差；
3. 对两个方法执行配对 t 检验、Welch 双样本 t 检验和 Wilcoxon 符号秩检验。

默认输入由 ``evaluate_external.py`` 生成。脚本不会补齐或猜测任何实验数据；
输入缺失、种子不完整或字段不合法时会直接报错并返回非零退出码。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ("accuracy", "precision", "recall", "f1", "auc")
METRIC_ALIASES = {
    "accuracy": ("accuracy", "acc"),
    "precision": ("precision",),
    "recall": ("recall", "sensitivity"),
    "f1": ("f1",),
    "auc": ("auc",),
}

LOGGER = logging.getLogger("stats_test")


def configure_logging() -> None:
    """配置简洁且适合终端保存的中文日志。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_csv_list(raw: str) -> list[str]:
    """解析逗号分隔参数，并拒绝空列表和重复值。"""

    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("逗号分隔参数不能为空")
    if len(values) != len(set(values)):
        raise ValueError(f"逗号分隔参数包含重复值：{raw}")
    return values


def parse_seed_list(raw: str) -> list[int]:
    """解析随机种子列表。"""

    try:
        seeds = [int(value) for value in parse_csv_list(raw)]
    except ValueError as error:
        raise ValueError(f"无法解析随机种子列表：{raw}") from error
    return seeds


def validate_metrics(metrics: Iterable[str]) -> tuple[str, ...]:
    """验证指标名称并保持用户给定顺序。"""

    normalized = tuple(str(metric).strip().lower() for metric in metrics)
    unsupported = [metric for metric in normalized if metric not in DEFAULT_METRICS]
    if unsupported:
        raise ValueError(
            f"不支持的指标：{unsupported}；可选值为 {list(DEFAULT_METRICS)}"
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError("指标列表不能包含重复值")
    return normalized


def _validate_binary_arrays(
    labels: Iterable[int],
    probabilities: Iterable[float],
    predictions: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将二分类输入转换为一维数组，并执行严格校验。"""

    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("标签、概率和预测类别必须是一维数组")
    if not (len(y_true) == len(y_score) == len(y_pred)) or len(y_true) == 0:
        raise ValueError("标签、概率和预测类别必须非空且长度一致")
    if not np.isfinite(y_score).all() or np.any((y_score < 0) | (y_score > 1)):
        raise ValueError("预测概率必须是 [0, 1] 内的有限数值")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("标签必须为 0/1")
    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError("预测类别必须为 0/1")
    return y_true, y_score, y_pred


def calculate_binary_metrics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    predictions: Iterable[int],
) -> dict[str, float]:
    """计算任务要求的五项二分类指标。

    AUC 必须同时包含正、负两类。Precision、Recall 和 F1 在分母为零时
    按 scikit-learn 约定返回 0，避免 Bootstrap 中产生无意义的异常。
    """

    y_true, y_score, y_pred = _validate_binary_arrays(
        labels, probabilities, predictions
    )
    if len(np.unique(y_true)) != 2:
        raise ValueError("AUC 计算要求样本同时包含正类和负类")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
    }


def prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """规范预测表，并在必要时由冻结阈值生成预测类别。"""

    frame = predictions.copy()
    required = {"method", "seed", "seriesuid", "label", "probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"预测表缺少必要字段：{missing}")
    if frame.empty:
        raise ValueError("预测表为空")
    if "dataset" not in frame.columns:
        frame["dataset"] = "fixed_test"
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["probability"] = pd.to_numeric(
        frame["probability"], errors="raise"
    ).astype(float)
    frame["seriesuid"] = frame["seriesuid"].astype(str)
    frame["method"] = frame["method"].astype(str)
    frame["dataset"] = frame["dataset"].astype(str)
    if frame["seriesuid"].eq("").any():
        raise ValueError("seriesuid 不能为空")

    if "prediction" not in frame.columns:
        if "threshold" not in frame.columns:
            raise ValueError("预测表必须包含 prediction，或包含可生成它的 threshold")
        frame["threshold"] = pd.to_numeric(
            frame["threshold"], errors="raise"
        ).astype(float)
        frame["prediction"] = (
            frame["probability"] >= frame["threshold"]
        ).astype(int)
    else:
        frame["prediction"] = pd.to_numeric(
            frame["prediction"], errors="raise"
        ).astype(int)

    # 先在全表层面校验取值；每个运行是否包含双类别在 Bootstrap 时单独检查。
    _validate_binary_arrays(
        frame["label"], frame["probability"], frame["prediction"]
    )
    return frame


def calculate_bootstrap_ci(
    predictions: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    cluster_col: str = "seriesuid",
    random_seed: int = 42,
) -> pd.DataFrame:
    """对单个模型运行执行 CT 簇 Bootstrap。

    每次从唯一 CT 列表中有放回抽取相同数量的 CT；某个 CT 被抽中多次时，
    其全部候选也会重复拼入本次样本。若重采样后只剩单类别，则该次不计入
    有效次数，并继续抽样，直到获得 ``n_bootstrap`` 次有效结果。
    """

    selected_metrics = validate_metrics(metrics)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap 必须为正整数")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level 必须位于 (0, 1)")
    frame = prepare_predictions(predictions)
    if cluster_col not in frame.columns:
        raise ValueError(f"预测表缺少簇字段：{cluster_col}")

    clusters = frame[cluster_col].astype(str).drop_duplicates().to_numpy()
    if len(clusters) < 2:
        raise ValueError("CT 簇 Bootstrap 至少需要 2 个不同的 seriesuid")
    cluster_indices = {
        cluster: np.flatnonzero(frame[cluster_col].astype(str).to_numpy() == cluster)
        for cluster in clusters
    }
    point = calculate_binary_metrics(
        frame["label"], frame["probability"], frame["prediction"]
    )

    rng = np.random.default_rng(random_seed)
    samples: dict[str, list[float]] = {metric: [] for metric in selected_metrics}
    attempts = 0
    maximum_attempts = max(n_bootstrap * 20, n_bootstrap + 100)
    while len(samples[selected_metrics[0]]) < n_bootstrap and attempts < maximum_attempts:
        attempts += 1
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_indices = np.concatenate(
            [cluster_indices[str(cluster)] for cluster in sampled_clusters]
        )
        sampled = frame.iloc[sampled_indices]
        try:
            values = calculate_binary_metrics(
                sampled["label"], sampled["probability"], sampled["prediction"]
            )
        except ValueError as error:
            if "同时包含正类和负类" in str(error):
                continue
            raise
        for metric in selected_metrics:
            samples[metric].append(values[metric])

    valid_count = len(samples[selected_metrics[0]])
    if valid_count != n_bootstrap:
        raise RuntimeError(
            f"在 {attempts} 次尝试后仅获得 {valid_count}/{n_bootstrap} "
            "次包含双类别的有效 Bootstrap 样本"
        )

    tail = (1.0 - confidence_level) / 2.0
    rows = []
    for metric in selected_metrics:
        distribution = np.asarray(samples[metric], dtype=np.float64)
        rows.append(
            {
                "metric": metric,
                "value": point[metric],
                "ci_lower": float(np.quantile(distribution, tail)),
                "ci_upper": float(np.quantile(distribution, 1.0 - tail)),
                "confidence_level": float(confidence_level),
                "bootstrap_iterations": int(n_bootstrap),
                "bootstrap_valid": int(valid_count),
                "bootstrap_attempts": int(attempts),
                "resample_unit": cluster_col,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_grouped_predictions(
    predictions: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    cluster_col: str = "seriesuid",
    random_seed: int = 42,
) -> pd.DataFrame:
    """按数据集、方法和随机种子分组计算 Bootstrap CI。"""

    frame = prepare_predictions(predictions)
    rows = []
    for keys, group in frame.groupby(["dataset", "method", "seed"], sort=True):
        dataset, method, seed = keys
        LOGGER.info(
            "Bootstrap：dataset=%s，method=%s，seed=%s，候选=%d，CT=%d",
            dataset,
            method,
            seed,
            len(group),
            group[cluster_col].nunique(),
        )
        result = calculate_bootstrap_ci(
            group,
            metrics=metrics,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            cluster_col=cluster_col,
            # 每组重新使用同一 RNG，有利于相同测试集上的模型使用一致抽样序列。
            random_seed=random_seed,
        )
        result.insert(0, "seed", int(seed))
        result.insert(0, "method", str(method))
        result.insert(0, "dataset", str(dataset))
        if "threshold" in group.columns:
            thresholds = pd.to_numeric(group["threshold"], errors="raise").unique()
            if len(thresholds) != 1:
                raise ValueError(f"{method} seed={seed} 存在多个冻结阈值")
            result["threshold"] = float(thresholds[0])
        result["n_candidates"] = int(len(group))
        result["n_series"] = int(group[cluster_col].nunique())
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def normalize_seed_results(
    results: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """将长表或宽表种子结果统一为 dataset/method/seed/metric/value 长表。"""

    selected_metrics = validate_metrics(metrics)
    frame = results.copy()
    required = {"method", "seed"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"种子结果表缺少必要字段：{missing}")
    if frame.empty:
        raise ValueError("种子结果表为空")
    if "dataset" not in frame.columns:
        frame["dataset"] = "fixed_test"
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)

    if {"metric", "value"}.issubset(frame.columns):
        long_frame = frame[["dataset", "method", "seed", "metric", "value"]].copy()
        long_frame["metric"] = long_frame["metric"].astype(str).str.lower()
        long_frame = long_frame[long_frame["metric"].isin(selected_metrics)].copy()
    else:
        pieces = []
        for metric in selected_metrics:
            source = next(
                (name for name in METRIC_ALIASES[metric] if name in frame.columns),
                None,
            )
            if source is None:
                raise ValueError(f"宽表缺少指标字段：{metric}")
            piece = frame[["dataset", "method", "seed", source]].copy()
            piece = piece.rename(columns={source: "value"})
            piece["metric"] = metric
            pieces.append(piece)
        long_frame = pd.concat(pieces, ignore_index=True)

    if long_frame.empty:
        raise ValueError("种子结果表中没有请求的指标")
    long_frame["dataset"] = long_frame["dataset"].astype(str)
    long_frame["method"] = long_frame["method"].astype(str)
    long_frame["value"] = pd.to_numeric(long_frame["value"], errors="raise").astype(float)
    if not np.isfinite(long_frame["value"]).all():
        raise ValueError("种子结果包含非有限指标值")
    duplicate = long_frame.duplicated(
        ["dataset", "method", "seed", "metric"], keep=False
    )
    if duplicate.any():
        records = long_frame.loc[
            duplicate, ["dataset", "method", "seed", "metric"]
        ].to_dict("records")
        raise ValueError(f"种子结果存在重复记录：{records[:5]}")
    return long_frame.sort_values(
        ["dataset", "method", "metric", "seed"]
    ).reset_index(drop=True)


def _require_requested_seeds(
    frame: pd.DataFrame,
    seeds: Sequence[int],
    context: str,
) -> pd.DataFrame:
    """筛选并严格检查指定随机种子。"""

    requested = [int(seed) for seed in seeds]
    subset = frame[frame["seed"].isin(requested)].copy()
    present = sorted(subset["seed"].unique().tolist())
    missing = sorted(set(requested) - set(present))
    if missing:
        raise ValueError(f"{context} 缺少随机种子：{missing}")
    return subset


def calculate_multiseed_summary(
    results: pd.DataFrame,
    seeds: Sequence[int] = (0, 1, 2),
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """计算各方法在指定种子上的均值、样本标准差和展示字符串。"""

    long_frame = normalize_seed_results(results, metrics)
    requested = [int(seed) for seed in seeds]
    rows = []
    for keys, group in long_frame.groupby(
        ["dataset", "method", "metric"], sort=True
    ):
        dataset, method, metric = keys
        selected = _require_requested_seeds(
            group, requested, f"{dataset}/{method}/{metric}"
        ).sort_values("seed")
        values = selected["value"].to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "seeds": ",".join(str(seed) for seed in requested),
                "n_seeds": len(values),
                "mean": mean,
                "std": std,
                "mean_std": f"{mean:.6f} ± {std:.6f}",
            }
        )
    return pd.DataFrame(rows)


def _safe_wilcoxon(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float, str]:
    """执行 Wilcoxon，并显式处理所有配对差值均为零的边界情况。"""

    differences = values_a - values_b
    if np.allclose(differences, 0.0, rtol=0.0, atol=1e-15):
        return 0.0, 1.0, "所有配对差值均为零"
    result = stats.wilcoxon(
        values_a,
        values_b,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue), ""


def run_significance_tests(
    results: pd.DataFrame,
    method_a: str,
    method_b: str,
    seeds: Sequence[int] = (0, 1, 2),
    metrics: Sequence[str] = DEFAULT_METRICS,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """比较两个方法，并同时输出三种双侧显著性检验。"""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 必须位于 (0, 1)")
    if method_a == method_b:
        raise ValueError("待比较的两个方法不能相同")
    requested = [int(seed) for seed in seeds]
    if len(requested) < 2:
        raise ValueError("t 检验至少需要 2 个随机种子")
    long_frame = normalize_seed_results(results, metrics)
    available_methods = set(long_frame["method"])
    missing_methods = [
        method for method in (method_a, method_b) if method not in available_methods
    ]
    if missing_methods:
        raise ValueError(f"种子结果缺少待比较方法：{missing_methods}")

    rows = []
    for dataset in sorted(long_frame["dataset"].unique()):
        dataset_frame = long_frame[long_frame["dataset"] == dataset]
        for metric in validate_metrics(metrics):
            a = dataset_frame[
                (dataset_frame["method"] == method_a)
                & (dataset_frame["metric"] == metric)
            ]
            b = dataset_frame[
                (dataset_frame["method"] == method_b)
                & (dataset_frame["metric"] == metric)
            ]
            a = _require_requested_seeds(
                a, requested, f"{dataset}/{method_a}/{metric}"
            ).set_index("seed")
            b = _require_requested_seeds(
                b, requested, f"{dataset}/{method_b}/{metric}"
            ).set_index("seed")
            values_a = a.loc[requested, "value"].to_numpy(dtype=np.float64)
            values_b = b.loc[requested, "value"].to_numpy(dtype=np.float64)

            paired_t = stats.ttest_rel(values_a, values_b, nan_policy="raise")
            welch_t = stats.ttest_ind(
                values_a,
                values_b,
                equal_var=False,
                nan_policy="raise",
                alternative="two-sided",
            )
            wilcoxon_stat, wilcoxon_p, wilcoxon_note = _safe_wilcoxon(
                values_a, values_b
            )
            tests = [
                (
                    "paired_t_test",
                    True,
                    float(paired_t.statistic),
                    float(paired_t.pvalue),
                    "",
                ),
                (
                    "welch_t_test",
                    False,
                    float(welch_t.statistic),
                    float(welch_t.pvalue),
                    "",
                ),
                (
                    "wilcoxon_signed_rank",
                    True,
                    wilcoxon_stat,
                    wilcoxon_p,
                    wilcoxon_note,
                ),
            ]
            for test_name, paired, statistic, p_value, note in tests:
                small_sample_note = f"n={len(requested)}，小样本结果需谨慎解释"
                combined_note = "；".join(
                    item for item in (note, small_sample_note) if item
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method_a": method_a,
                        "method_b": method_b,
                        "metric": metric,
                        "test": test_name,
                        "paired": paired,
                        "n_a": len(values_a),
                        "n_b": len(values_b),
                        "statistic": statistic,
                        "p_value": p_value,
                        "alpha": float(alpha),
                        "significant": bool(np.isfinite(p_value) and p_value < alpha),
                        "note": combined_note,
                    }
                )
    return pd.DataFrame(rows)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """以适合中文表格软件读取的 UTF-8 BOM 编码保存 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    LOGGER.info("已保存 %s（%d 行）", path, len(frame))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="肺结节分类实验统计分析")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "runs/external_test/predictions.csv",
        help="逐候选预测 CSV",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "runs/external_test/metrics_with_ci.csv",
        help="逐 seed 指标 CSV，可为长表或宽表",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/stats",
    )
    parser.add_argument("--method-a", default="resnet18_augmented")
    parser.add_argument("--method-b", default="pbip_lite")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args(argv)


def run_analysis(args: argparse.Namespace) -> dict[str, Path]:
    """执行完整统计流程并返回输出文件路径。"""

    if not args.predictions.exists():
        raise FileNotFoundError(f"预测文件不存在：{args.predictions}")
    if not args.results.exists():
        raise FileNotFoundError(f"种子结果文件不存在：{args.results}")
    seeds = parse_seed_list(args.seeds)
    metrics = validate_metrics(parse_csv_list(args.metrics))
    predictions = pd.read_csv(args.predictions)
    results = pd.read_csv(args.results)

    bootstrap = bootstrap_grouped_predictions(
        predictions,
        metrics=metrics,
        n_bootstrap=args.bootstrap_iterations,
        confidence_level=args.confidence_level,
        cluster_col="seriesuid",
        random_seed=args.random_seed,
    )
    summary = calculate_multiseed_summary(results, seeds=seeds, metrics=metrics)
    significance = run_significance_tests(
        results,
        method_a=args.method_a,
        method_b=args.method_b,
        seeds=seeds,
        metrics=metrics,
        alpha=args.alpha,
    )

    output_dir = args.output_dir.resolve()
    paths = {
        "bootstrap": output_dir / "bootstrap_ci.csv",
        "summary": output_dir / "multiseed_summary.csv",
        "significance": output_dir / "significance_tests.csv",
    }
    save_csv(bootstrap, paths["bootstrap"])
    save_csv(summary, paths["summary"])
    save_csv(significance, paths["significance"])

    LOGGER.info("多种子汇总：\n%s", summary.to_string(index=False))
    LOGGER.info("显著性检验：\n%s", significance.to_string(index=False))
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """命令行入口。"""

    configure_logging()
    args = parse_args(argv)
    run_analysis(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # CLI 需要记录完整堆栈并返回失败状态
        LOGGER.exception("统计分析失败：%s", error)
        sys.exit(1)
