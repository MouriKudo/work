"""LIDC-IDRI 优先、LUNA16 固定测试集回退的统一评估入口。

脚本默认评估强增强 ResNet18 与 PBIP-Lite 的 seed 0/1/2。LIDC-IDRI
只有在完成去重、与 LUNA16 无重叠、包含合法二分类标签且 patch 完整时才会
进入正式评估；否则 ``auto`` 模式会记录原因并回退到冻结的 LUNA16 test split。

分类阈值始终只在 LUNA16 validation split 上按最大 F1 选择，任何测试数据均不
参与阈值调优。输出预测和指标全部来自真实模型推理，不生成占位数据。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_utils import METHOD_DIRS, make_loader, run_spec_from_dir
from metrics import binary_metrics, collect_predictions, find_best_f1_threshold, load_model
from stats_test import (
    DEFAULT_METRICS,
    bootstrap_grouped_predictions,
    parse_csv_list,
    parse_seed_list,
    save_csv,
)


LOGGER = logging.getLogger("evaluate_external")


@dataclass(frozen=True)
class EvaluationSource:
    """一个通过完整性检查、可直接构建 DataLoader 的评估数据源。"""

    name: str
    metadata: Path
    patches: Path
    split: str
    origin_format: str


def configure_logging(log_file: Path | None = None) -> None:
    """配置终端与 UTF-8 文件日志，确保评估过程可独立归档。"""

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8", mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _read_metadata(path: Path, source_name: str) -> tuple[pd.DataFrame | None, list[str]]:
    """安全读取元数据，并把错误转换为数据源不可用原因。"""

    if not path.exists():
        return None, [f"{source_name} 元数据不存在：{path}"]
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        return None, [f"{source_name} 元数据读取失败：{error!r}"]
    if frame.empty:
        return frame, [f"{source_name} 元数据为空"]
    return frame, []


def _check_common_dataset(
    frame: pd.DataFrame,
    patches_dir: Path,
    split: str,
    source_name: str,
) -> tuple[list[str], dict[str, object]]:
    """检查两类数据源共同需要的字段、类别、CT 和 patch 文件。"""

    reasons: list[str] = []
    required = {"seriesuid", "patch_file", "class", "split"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        return [f"{source_name} 元数据缺少字段：{missing_columns}"], {}

    selected = frame[frame["split"].astype(str) == split].copy()
    if selected.empty:
        reasons.append(f"{source_name} 中不存在 split={split} 的样本")
        return reasons, {
            "rows": 0,
            "series": 0,
            "positive": 0,
            "negative": 0,
            "classes": [],
            "missing_patches": 0,
        }

    try:
        selected["class"] = pd.to_numeric(selected["class"], errors="raise").astype(int)
    except Exception as error:
        reasons.append(f"{source_name} class 字段无法解析为整数：{error!r}")
        return reasons, {}

    classes = sorted(selected["class"].unique().tolist())
    if classes != [0, 1]:
        reasons.append(
            f"{source_name} 不满足二分类评估要求，实际类别={classes}，必须同时包含 0/1"
        )
    series_count = int(selected["seriesuid"].astype(str).nunique())
    if series_count < 2:
        reasons.append(f"{source_name} 仅有 {series_count} 个 CT，无法执行 CT 簇 Bootstrap")

    duplicated_patches = int(selected["patch_file"].astype(str).duplicated().sum())
    if duplicated_patches:
        reasons.append(f"{source_name} 含 {duplicated_patches} 条重复 patch_file")
    missing_patch_paths = [
        str(patches_dir / str(name))
        for name in selected["patch_file"]
        if not (patches_dir / str(name)).is_file()
    ]
    if missing_patch_paths:
        reasons.append(
            f"{source_name} 缺少 {len(missing_patch_paths)} 个 patch 文件，"
            f"首个缺失项={missing_patch_paths[0]}"
        )

    summary = {
        "rows": int(len(selected)),
        "series": series_count,
        "positive": int((selected["class"] == 1).sum()),
        "negative": int((selected["class"] == 0).sum()),
        "classes": classes,
        "missing_patches": len(missing_patch_paths),
        "duplicated_patch_files": duplicated_patches,
    }
    return reasons, summary


def inspect_lidc_readiness(
    metadata_path: Path,
    patches_dir: Path,
    dedup_stats_path: Path,
    luna_metadata_path: Path,
) -> dict[str, object]:
    """判断 LIDC-IDRI 是否具备正式二分类评估条件。"""

    reasons: list[str] = []
    dedup_stats: dict[str, object] = {}
    if not dedup_stats_path.exists():
        reasons.append(f"LIDC 去重统计不存在：{dedup_stats_path}")
    else:
        try:
            dedup_stats = json.loads(dedup_stats_path.read_text(encoding="utf-8-sig"))
            if int(dedup_stats.get("xml_unique_series", 0)) <= 0:
                reasons.append("LIDC XML 去重后没有唯一 SeriesUID")
            if int(dedup_stats.get("failed_patches", 0)) > 0:
                reasons.append(
                    f"LIDC patch 提取仍有 {dedup_stats['failed_patches']} 条失败记录"
                )
        except Exception as error:
            reasons.append(f"LIDC 去重统计读取失败：{error!r}")

    frame, read_reasons = _read_metadata(metadata_path, "LIDC-IDRI")
    reasons.extend(read_reasons)
    summary: dict[str, object] = {}
    if frame is not None and not frame.empty:
        common_reasons, summary = _check_common_dataset(
            frame, patches_dir, "external", "LIDC-IDRI"
        )
        reasons.extend(common_reasons)

        # 使用 SeriesUID 与全部 LUNA16 元数据再次交叉核验，而不是只相信旧清单。
        if {"seriesuid", "split"}.issubset(frame.columns) and luna_metadata_path.exists():
            try:
                luna_frame = pd.read_csv(luna_metadata_path, usecols=["seriesuid"])
                external_uids = set(
                    frame.loc[
                        frame["split"].astype(str) == "external", "seriesuid"
                    ].astype(str)
                )
                overlap = external_uids & set(luna_frame["seriesuid"].astype(str))
                summary["luna16_overlap_series"] = len(overlap)
                if overlap:
                    reasons.append(
                        f"LIDC 评估元数据仍有 {len(overlap)} 个 SeriesUID 与 LUNA16 重叠"
                    )
            except Exception as error:
                reasons.append(f"LIDC/LUNA16 UID 交叉核验失败：{error!r}")
        elif not luna_metadata_path.exists():
            reasons.append(f"无法执行 LIDC/LUNA16 UID 核验：{luna_metadata_path} 不存在")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "summary": summary,
        "deduplication": dedup_stats,
    }


def inspect_luna_readiness(
    metadata_path: Path,
    patches_dir: Path,
) -> dict[str, object]:
    """检查冻结的 LUNA16 test split 是否完整。"""

    frame, reasons = _read_metadata(metadata_path, "LUNA16")
    summary: dict[str, object] = {}
    if frame is not None and not frame.empty:
        common_reasons, summary = _check_common_dataset(
            frame, patches_dir, "test", "LUNA16"
        )
        reasons.extend(common_reasons)
    return {"ready": not reasons, "reasons": reasons, "summary": summary}


def choose_evaluation_source(
    mode: str,
    lidc_metadata: Path,
    lidc_patches: Path,
    lidc_dedup_stats: Path,
    luna_metadata: Path,
    luna_patches: Path,
) -> tuple[EvaluationSource, dict[str, object]]:
    """按指定模式选择评估数据源，并返回完整审计信息。"""

    if mode not in {"auto", "lidc", "luna16"}:
        raise ValueError(f"未知数据源模式：{mode}")
    lidc_audit = inspect_lidc_readiness(
        lidc_metadata,
        lidc_patches,
        lidc_dedup_stats,
        luna_metadata,
    )
    luna_audit = inspect_luna_readiness(luna_metadata, luna_patches)
    audit = {
        "requested_mode": mode,
        "lidc": lidc_audit,
        "luna16": luna_audit,
        "fallback_used": False,
        "fallback_reasons": [],
    }

    if mode in {"auto", "lidc"} and bool(lidc_audit["ready"]):
        return (
            EvaluationSource(
                name="lidc_idri_external",
                metadata=lidc_metadata.resolve(),
                patches=lidc_patches.resolve(),
                split="external",
                origin_format="DICOM/XML（已提取 3×64×64 NPY patch）",
            ),
            audit,
        )
    if mode == "lidc":
        raise RuntimeError(
            "强制 LIDC-IDRI 评估失败：" + "；".join(lidc_audit["reasons"])
        )
    if not bool(luna_audit["ready"]):
        raise RuntimeError(
            "LUNA16 固定测试集不可用：" + "；".join(luna_audit["reasons"])
        )
    if mode == "auto":
        audit["fallback_used"] = True
        audit["fallback_reasons"] = list(lidc_audit["reasons"])
    return (
        EvaluationSource(
            name="luna16_fixed_test",
            metadata=luna_metadata.resolve(),
            patches=luna_patches.resolve(),
            split="test",
            origin_format="MHD（已提取 3×64×64 NPY patch）",
        ),
        audit,
    )


def resolve_device(raw: str) -> torch.device:
    """解析计算设备参数。"""

    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 PyTorch 未检测到可用 GPU")
    return torch.device(raw)


def discover_requested_runs(
    runs_root: Path,
    methods: Sequence[str],
    seeds: Sequence[int],
):
    """只发现用户请求的方法和种子，避免无关实验缺失阻断评估。"""

    unsupported = [method for method in methods if method not in METHOD_DIRS]
    if unsupported:
        raise ValueError(
            f"不支持的方法：{unsupported}；可选值为 {list(METHOD_DIRS)}"
        )
    specs = []
    for method in methods:
        for seed in seeds:
            run_dir = runs_root / f"seed_{seed}" / METHOD_DIRS[method]
            specs.append(run_spec_from_dir(method, int(seed), run_dir))
    return specs


def _loader_for_source(
    source: EvaluationSource,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[object, DataLoader]:
    return make_loader(
        source.split,
        source.metadata,
        source.patches,
        batch_size,
        num_workers,
        device,
    )


@torch.no_grad()
def evaluate_runs(
    specs,
    source: EvaluationSource,
    luna_metadata: Path,
    luna_patches: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐模型执行验证集阈值冻结和目标数据推理。"""

    _, validation_loader = make_loader(
        "val",
        luna_metadata,
        luna_patches,
        batch_size,
        num_workers,
        device,
    )
    source_dataset, source_loader = _loader_for_source(
        source, batch_size, num_workers, device
    )
    prediction_frames = []
    run_rows = []

    for spec in specs:
        LOGGER.info(
            "加载模型：method=%s，seed=%d，checkpoint=%s",
            spec.method,
            spec.seed,
            spec.checkpoint,
        )
        model = load_model(spec.model_type, spec.checkpoint, spec.prototype_bank).to(device)
        val_probabilities, val_labels, _ = collect_predictions(
            model, validation_loader, device
        )
        threshold, validation_f1 = find_best_f1_threshold(
            val_labels, val_probabilities
        )
        probabilities, labels, seriesuids = collect_predictions(
            model, source_loader, device
        )
        predicted = (probabilities >= threshold).astype(np.int64)
        metrics = binary_metrics(labels, probabilities, threshold)
        LOGGER.info(
            "评估完成：method=%s，seed=%d，threshold=%.6f，Acc=%.4f，"
            "Precision=%.4f，Recall=%.4f，F1=%.4f，AUC=%.4f",
            spec.method,
            spec.seed,
            threshold,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["auc"],
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "dataset": source.name,
                    "method": spec.method,
                    "seed": int(spec.seed),
                    "sample_index": np.arange(len(labels), dtype=np.int64),
                    "seriesuid": seriesuids.astype(str),
                    "label": labels.astype(np.int64),
                    "probability": probabilities.astype(np.float64),
                    "threshold": float(threshold),
                    "prediction": predicted,
                }
            )
        )
        run_rows.append(
            {
                "dataset": source.name,
                "method": spec.method,
                "seed": int(spec.seed),
                "checkpoint": str(spec.checkpoint.resolve()),
                "prototype_bank": (
                    str(spec.prototype_bank.resolve()) if spec.prototype_bank else ""
                ),
                "best_val_auc": float(spec.best_val_auc),
                "threshold": float(threshold),
                "validation_f1": float(validation_f1),
                "n_candidates": int(len(source_dataset)),
                "n_series": int(len(np.unique(seriesuids))),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(run_rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIDC-IDRI/LUNA16 外部或固定测试评估")
    parser.add_argument(
        "--data-source",
        choices=["auto", "lidc", "luna16"],
        default="auto",
    )
    parser.add_argument(
        "--methods",
        default="resnet18_augmented,pbip_lite",
        help="逗号分隔的方法列表",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs/experiments_v2",
    )
    parser.add_argument(
        "--luna-metadata",
        type=Path,
        default=PROJECT_ROOT / "data/processed/metadata.csv",
    )
    parser.add_argument(
        "--luna-patches",
        type=Path,
        default=PROJECT_ROOT / "data/processed/patches",
    )
    parser.add_argument(
        "--lidc-metadata",
        type=Path,
        default=PROJECT_ROOT / "runs/external_validation/lidc_external_metadata.csv",
    )
    parser.add_argument(
        "--lidc-patches",
        type=Path,
        default=PROJECT_ROOT / "data/processed/lidc_external_patches",
    )
    parser.add_argument(
        "--lidc-dedup-stats",
        type=Path,
        default=PROJECT_ROOT / "runs/external_validation/deduplication_stats.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/external_test",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="评估日志路径；默认保存为 output-dir/evaluation.log",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args(argv)


def run_evaluation(args: argparse.Namespace) -> dict[str, Path]:
    """执行数据源选择、模型推理、Bootstrap 和结果落盘。"""

    if args.batch_size <= 0:
        raise ValueError("batch-size 必须为正整数")
    if args.num_workers < 0:
        raise ValueError("num-workers 不能为负数")
    methods = parse_csv_list(args.methods)
    seeds = parse_seed_list(args.seeds)
    source, audit = choose_evaluation_source(
        args.data_source,
        args.lidc_metadata,
        args.lidc_patches,
        args.lidc_dedup_stats,
        args.luna_metadata,
        args.luna_patches,
    )
    if audit["fallback_used"]:
        LOGGER.warning(
            "LIDC-IDRI 不满足正式评估条件，自动回退 LUNA16 固定测试集：%s",
            "；".join(audit["fallback_reasons"]),
        )
    LOGGER.info("最终评估数据源：%s（%s）", source.name, source.metadata)

    device = resolve_device(args.device)
    LOGGER.info("计算设备：%s", device)
    specs = discover_requested_runs(args.runs_root, methods, seeds)
    predictions, run_records = evaluate_runs(
        specs,
        source,
        args.luna_metadata,
        args.luna_patches,
        args.batch_size,
        args.num_workers,
        device,
    )
    metrics = bootstrap_grouped_predictions(
        predictions,
        metrics=DEFAULT_METRICS,
        n_bootstrap=args.bootstrap_iterations,
        confidence_level=args.confidence_level,
        cluster_col="seriesuid",
        random_seed=args.bootstrap_seed,
    )
    metrics = metrics.merge(
        run_records.drop(columns=["threshold", "n_candidates", "n_series"]),
        on=["dataset", "method", "seed"],
        how="left",
        validate="many_to_one",
    )

    # 各模型评估的是同一测试集，只取第一组预测统计真实候选类别数，避免按
    # 方法/seed 重复计数，也不能按 seriesuid 去重（同一 CT 可含多个候选）。
    reference_predictions = predictions[
        (predictions["method"] == methods[0])
        & (predictions["seed"] == seeds[0])
    ]
    dataset_group = reference_predictions.groupby("label").size()
    manifest = {
        "selected_source": asdict(source),
        "source_audit": audit,
        "threshold_protocol": "每个模型仅在 LUNA16 validation split 最大化 F1",
        "evaluation_protocol": "候选级二分类；按 seriesuid 执行 CT 簇 Bootstrap",
        "methods": methods,
        "seeds": seeds,
        "device": str(device),
        "bootstrap": {
            "iterations": int(args.bootstrap_iterations),
            "confidence_level": float(args.confidence_level),
            "random_seed": int(args.bootstrap_seed),
            "resample_unit": "seriesuid",
        },
        "evaluated_candidates": int(
            len(predictions) // max(1, len(methods) * len(seeds))
        ),
        "evaluated_series": int(predictions["seriesuid"].nunique()),
        "class_counts": {
            str(int(label)): int(count) for label, count in dataset_group.items()
        },
    }
    # Path 不能直接 JSON 序列化，转换为稳定字符串。
    manifest["selected_source"]["metadata"] = str(source.metadata)
    manifest["selected_source"]["patches"] = str(source.patches)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_dir / "data_source_manifest.json",
        "predictions": output_dir / "predictions.csv",
        "metrics": output_dir / "metrics_with_ci.csv",
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_csv(predictions, paths["predictions"])
    save_csv(metrics, paths["metrics"])
    LOGGER.info("指标结果：\n%s", metrics.to_string(index=False))
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """命令行入口。"""

    args = parse_args(argv)
    log_file = args.log_file or args.output_dir / "evaluation.log"
    configure_logging(log_file.resolve())
    run_evaluation(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # CLI 失败时保留完整堆栈，便于复现实验
        LOGGER.exception("外部/固定测试评估失败：%s", error)
        sys.exit(1)
