"""肺结节分类工程的一键冒烟测试。

该脚本使用现有 LUNA16 小样本和正式 checkpoint，依次验证：

1. SimpleITK 能够读取 ``.mhd``；
2. Dataset/DataLoader 能够加载三层 patch；
3. PBIP-Lite 或 ResNet18 checkpoint 能够完成推理；
4. 能够计算分类指标与 CT 簇 Bootstrap 置信区间；
5. 能够导出 CSV、JSON、日志和诊断图。

冒烟测试只验证工程链路，不替代完整固定测试或外部验证。测试阈值由抽取的
验证小样本确定，因此输出指标不得作为论文主结果引用。
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from luna16_dataset import LUNA16Dataset  # noqa: E402
from metrics import (  # noqa: E402
    binary_metrics,
    collect_predictions,
    find_best_f1_threshold,
    load_model,
)
from plot_utils import render_line_chart  # noqa: E402
from stats_test import calculate_bootstrap_ci  # noqa: E402


LOGGER = logging.getLogger("smoke_test")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数，并提供当前正式 PBIP 产物作为默认值。"""

    parser = argparse.ArgumentParser(description="肺结节分类工程一键冒烟测试")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data/processed/metadata.csv",
    )
    parser.add_argument(
        "--patches",
        type=Path,
        default=PROJECT_ROOT / "data/processed/patches",
    )
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument(
        "--model-type", choices=["pbip", "resnet18"], default="pbip"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "runs/experiments_v2/seed_1/pbip_contrast/best_model.pth"
        ),
    )
    parser.add_argument(
        "--prototype-bank",
        type=Path,
        default=(
            PROJECT_ROOT
            / "runs/experiments_v2/seed_1/prototype_bank/prototype_bank.pkl"
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=32,
        help="验证集和测试集分别抽取的平衡候选数，必须为不小于 8 的偶数",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/smoke_test/latest",
    )
    parser.add_argument(
        "--skip-mhd-check",
        action="store_true",
        help="仅在原始 .mhd 不可用的精简部署环境中跳过格式读取检查",
    )
    return parser.parse_args(argv)


def configure_logging(log_path: Path) -> None:
    """同时记录终端和 UTF-8 日志文件。"""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    LOGGER.propagate = False


def resolve_device(requested: str) -> torch.device:
    """根据参数和 CUDA 可用性选择运行设备。"""

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境没有可用 CUDA")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_path(path: Path, kind: str) -> Path:
    """验证输入文件或目录，避免静默使用错误路径。"""

    resolved = path.resolve()
    if kind == "file" and not resolved.is_file():
        raise FileNotFoundError(f"找不到文件：{resolved}")
    if kind == "dir" and not resolved.is_dir():
        raise FileNotFoundError(f"找不到目录：{resolved}")
    return resolved


def verify_mhd_readable(raw_dir: Path) -> dict[str, Any]:
    """用 SimpleITK 读取一例真实 MHD，并返回最小空间信息。"""

    candidates = sorted(raw_dir.glob("subset*/**/*.mhd"))
    if not candidates:
        candidates = sorted(raw_dir.rglob("*.mhd"))
    if not candidates:
        raise FileNotFoundError(f"{raw_dir} 下没有可供冒烟测试的 .mhd")
    path = candidates[0]
    image = sitk.ReadImage(str(path))
    return {
        "path": str(path.resolve()),
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": [float(value) for value in image.GetSpacing()],
        "pixel_id": image.GetPixelIDTypeAsString(),
    }


def select_balanced_indices(
    dataset: LUNA16Dataset,
    sample_size: int,
    random_seed: int,
) -> tuple[list[int], pd.DataFrame]:
    """按类别平衡抽样，并优先让每个候选来自不同 CT。"""

    if sample_size < 8 or sample_size % 2:
        raise ValueError("--sample-size 必须为不小于 8 的偶数")
    selected_parts: list[pd.DataFrame] = []
    per_class = sample_size // 2
    for class_id in (0, 1):
        group = dataset.df[dataset.df["class"] == class_id]
        if len(group) < per_class:
            raise ValueError(
                f"split={dataset.df['split'].iloc[0]} 的 class={class_id} "
                f"只有 {len(group)} 个样本，无法抽取 {per_class} 个"
            )
        unique_ct = group.drop_duplicates("seriesuid")
        source = unique_ct if len(unique_ct) >= per_class else group
        sampled = source.sample(n=per_class, random_state=random_seed + class_id)
        selected_parts.append(sampled)

    selected = (
        pd.concat(selected_parts)
        .sample(frac=1.0, random_state=random_seed + 100)
        .copy()
    )
    indices = selected.index.astype(int).tolist()
    return indices, selected.reset_index(drop=True)


def create_small_loader(
    dataset: LUNA16Dataset,
    sample_size: int,
    random_seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, pd.DataFrame]:
    """创建固定、无随机增强的小样本 DataLoader。"""

    indices, frame = select_balanced_indices(dataset, sample_size, random_seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=min(batch_size, sample_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return loader, frame


def save_dashboard(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """使用 Pillow 保存 ROC、PR、概率分布和混淆矩阵诊断图。

    本项目保留了 matplotlib 论文绘图脚本；冒烟测试采用仓库内的 Pillow
    轻量绘图后端，是为了规避少数 Windows/CUDA 环境中 matplotlib 在完成
    torch 推理后保存图像时出现的原生 DLL 冲突。
    """

    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])

    roc_image = render_line_chart(
        {"ROC": (fpr, tpr), "chance": ([0.0, 1.0], [0.0, 1.0])},
        "ROC curve",
        "False positive rate",
        "True positive rate",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        width=620,
        height=420,
    )
    pr_image = render_line_chart(
        {"PR": (recall, precision)},
        "Precision-Recall curve",
        "Recall",
        "Precision",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        width=620,
        height=420,
    )

    ecdf_series = {}
    for class_id in (0, 1):
        values = np.sort(probabilities[labels == class_id])
        cumulative = np.arange(1, len(values) + 1, dtype=float) / len(values)
        ecdf_series[f"class={class_id}"] = (values, cumulative)
    ecdf_image = render_line_chart(
        ecdf_series,
        f"Probability ECDF (threshold={threshold:.3f})",
        "Nodule probability",
        "Cumulative fraction",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        width=620,
        height=420,
    )

    cm_image = Image.new("RGB", (620, 420), "white")
    draw = ImageDraw.Draw(cm_image)
    font = ImageFont.load_default()
    draw.text((72, 18), "Confusion matrix", fill="black", font=font)
    cell = 120
    left, top = 180, 90
    maximum = max(1, int(cm.max()))
    for row in range(2):
        for column in range(2):
            value = int(cm[row, column])
            intensity = int(245 - 150 * value / maximum)
            fill = (intensity, intensity + 5, 255)
            box = (
                left + column * cell,
                top + row * cell,
                left + (column + 1) * cell,
                top + (row + 1) * cell,
            )
            draw.rectangle(box, fill=fill, outline="#2F5597", width=2)
            draw.text((box[0] + 54, box[1] + 52), str(value), fill="black", font=font)
    draw.text((left + 35, top + 2 * cell + 12), "Pred 0", fill="black", font=font)
    draw.text((left + cell + 35, top + 2 * cell + 12), "Pred 1", fill="black", font=font)
    draw.text((left - 55, top + 55), "True 0", fill="black", font=font)
    draw.text((left - 55, top + cell + 55), "True 1", fill="black", font=font)

    dashboard = Image.new("RGB", (1240, 840), "white")
    dashboard.paste(roc_image, (0, 0))
    dashboard.paste(pr_image, (620, 0))
    dashboard.paste(ecdf_image, (0, 420))
    dashboard.paste(cm_image, (620, 420))
    dashboard.save(output_path, dpi=(300, 300))


def json_ready(value: Any) -> Any:
    """把 NumPy/Path 等对象转换为 JSON 可序列化类型。"""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_smoke_test(args: argparse.Namespace) -> dict[str, Any]:
    """执行加载、推理、评估、统计和绘图五阶段冒烟测试。"""

    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "smoke_test.log")
    steps: list[dict[str, Any]] = []

    def record_step(name: str, step_started: float, detail: str) -> None:
        elapsed = time.perf_counter() - step_started
        steps.append(
            {"step": name, "status": "PASS", "elapsed_seconds": round(elapsed, 4), "detail": detail}
        )
        LOGGER.info("PASS | %s | %.3fs | %s", name, elapsed, detail)

    LOGGER.info("开始一键冒烟测试，输出目录：%s", output_dir)
    metadata = require_path(args.metadata, "file")
    patches = require_path(args.patches, "dir")
    checkpoint = require_path(args.checkpoint, "file")
    prototype_bank = None
    if args.model_type == "pbip":
        prototype_bank = require_path(args.prototype_bank, "file")
    device = resolve_device(args.device)
    LOGGER.info("设备：%s；checkpoint：%s", device, checkpoint)

    step_started = time.perf_counter()
    mhd_info: dict[str, Any] | None = None
    if not args.skip_mhd_check:
        mhd_info = verify_mhd_readable(require_path(args.raw_dir, "dir"))
    val_dataset = LUNA16Dataset(metadata, patches, split="val")
    test_dataset = LUNA16Dataset(metadata, patches, split="test")
    val_loader, val_selected = create_small_loader(
        val_dataset, args.sample_size, args.random_seed, args.batch_size, device
    )
    test_loader, test_selected = create_small_loader(
        test_dataset,
        args.sample_size,
        args.random_seed + 1000,
        args.batch_size,
        device,
    )
    selected_samples = pd.concat(
        [val_selected.assign(smoke_split="validation"), test_selected.assign(smoke_split="test")],
        ignore_index=True,
    )
    selected_samples.to_csv(output_dir / "selected_samples.csv", index=False, encoding="utf-8")
    record_step(
        "加载",
        step_started,
        f"val={len(val_selected)}, test={len(test_selected)}, mhd_check={not args.skip_mhd_check}",
    )

    step_started = time.perf_counter()
    model = load_model(args.model_type, checkpoint, prototype_bank).to(device)
    val_probability, val_labels, _ = collect_predictions(model, val_loader, device)
    threshold, validation_f1 = find_best_f1_threshold(val_labels, val_probability)
    test_probability, test_labels, test_seriesuids = collect_predictions(
        model, test_loader, device
    )
    test_prediction = (test_probability >= threshold).astype(np.int64)
    record_step(
        "推理",
        step_started,
        f"model={args.model_type}, threshold={threshold:.6f}, validation_f1={validation_f1:.4f}",
    )

    step_started = time.perf_counter()
    metrics = binary_metrics(test_labels, test_probability, threshold)
    metric_rows = [
        {"metric": key, "value": value}
        for key, value in metrics.items()
        if key != "confusion_matrix" and isinstance(value, (int, float))
    ]
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8")
    predictions_frame = pd.DataFrame(
        {
            "dataset": "smoke_luna16_test_subset",
            "method": args.model_type,
            "seed": args.seed,
            "seriesuid": test_seriesuids,
            "label": test_labels.astype(int),
            "probability": test_probability.astype(float),
            "threshold": float(threshold),
            "prediction": test_prediction.astype(int),
        }
    )
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")
    record_step(
        "评估",
        step_started,
        f"AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}",
    )

    step_started = time.perf_counter()
    bootstrap = calculate_bootstrap_ci(
        predictions_frame,
        n_bootstrap=args.bootstrap_iterations,
        confidence_level=args.confidence_level,
        cluster_col="seriesuid",
        random_seed=args.random_seed,
    )
    bootstrap.to_csv(output_dir / "bootstrap_ci.csv", index=False, encoding="utf-8")
    record_step(
        "统计",
        step_started,
        f"CT 簇 Bootstrap={args.bootstrap_iterations}, confidence={args.confidence_level:.2f}",
    )

    step_started = time.perf_counter()
    save_dashboard(
        test_labels,
        test_probability,
        test_prediction,
        threshold,
        output_dir / "smoke_dashboard.png",
    )
    record_step("绘图", step_started, "smoke_dashboard.png, 300 DPI")

    pd.DataFrame(steps).to_csv(output_dir / "step_status.csv", index=False, encoding="utf-8")
    summary = {
        "status": "PASS",
        "purpose": "engineering_smoke_test_not_scientific_evaluation",
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "inputs": {
            "metadata": str(metadata),
            "patches": str(patches),
            "mhd": mhd_info,
            "model_type": args.model_type,
            "checkpoint": str(checkpoint),
            "prototype_bank": str(prototype_bank) if prototype_bank else None,
        },
        "sample": {
            "validation_candidates": len(val_selected),
            "test_candidates": len(test_selected),
            "test_series": int(len(np.unique(test_seriesuids))),
            "test_class_counts": {
                str(key): int(value)
                for key, value in zip(*np.unique(test_labels, return_counts=True))
            },
        },
        "threshold": float(threshold),
        "validation_f1": float(validation_f1),
        "metrics": metrics,
        "bootstrap_iterations": args.bootstrap_iterations,
        "steps": steps,
        "outputs": {
            "selected_samples": str(output_dir / "selected_samples.csv"),
            "predictions": str(output_dir / "predictions.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "bootstrap_ci": str(output_dir / "bootstrap_ci.csv"),
            "dashboard": str(output_dir / "smoke_dashboard.png"),
            "log": str(output_dir / "smoke_test.log"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("冒烟测试完成：PASS；总耗时 %.3fs", summary["elapsed_seconds"])
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """命令行入口；异常时保留日志并返回非零退出码。"""

    args = parse_args(argv)
    try:
        run_smoke_test(args)
    except Exception:
        if LOGGER.handlers:
            LOGGER.exception("冒烟测试失败")
        else:
            print("冒烟测试失败", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
