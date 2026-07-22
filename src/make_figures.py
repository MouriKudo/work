"""从真实实验 CSV、模型权重和医学影像 patch 生成论文图表。

默认输出 300 DPI PNG 和可缩放 SVG，覆盖项目流程、PBIP-Lite 结构、四方法
主结果、K 值消融、六类退化鲁棒性、Grad-CAM 四分类及 Top-3 病例检索。
脚本只读取已有实验产物；输入缺失时直接报错，不生成占位图或虚构数值。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_utils import best_run_per_method, discover_main_runs
from gradcam import GradCAM, overlay_cam
from metrics import load_model


LOGGER = logging.getLogger("make_figures")

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
    "pbip_full": "PBIP + 对比损失",
}
METHOD_COLORS = {
    "resnet18_baseline": "#7A5195",
    "resnet18_augmented": "#2F6B9A",
    "pbip_lite": "#2A9D8F",
    "pbip_full": "#E07A5F",
}
METRIC_COLORS = {"AUC": "#2F6B9A", "PR-AUC": "#E07A5F", "F1": "#2A9D8F"}
DEGRADATION_ORDER = (
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
    "window_shift",
    "jpeg",
    "resample",
)
DEGRADATION_LABELS_ZH = {
    "low_contrast": "低对比度",
    "gaussian_noise": "高斯噪声",
    "gaussian_blur": "高斯模糊",
    "window_shift": "窗位偏移",
    "jpeg": "JPEG 压缩",
    "resample": "下采样",
}
CATEGORY_LABELS_ZH = {
    "TP": "TP 真阳性",
    "FP": "FP 假阳性",
    "FN": "FN 假阴性",
    "TN": "TN 真阴性",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_academic_style() -> None:
    """配置适合中英文论文图的统一 Matplotlib 风格。"""

    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#D9E1E8",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            # 将中文字体转为 SVG 路径，避免在其他计算机上缺字体。
            "svg.fonttype": "path",
        }
    )


def parse_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    if not formats:
        raise ValueError("至少需要一种输出格式")
    unsupported = [fmt for fmt in formats if fmt not in {"png", "svg"}]
    if unsupported:
        raise ValueError(f"不支持的图片格式：{unsupported}；仅支持 PNG/SVG")
    if len(formats) != len(set(formats)):
        raise ValueError("输出格式不能重复")
    return formats


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} 缺少必要字段：{missing}")
    if frame.empty:
        raise ValueError(f"{source} 为空")


def read_csv_checked(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"绘图数据不存在：{path}")
    frame = pd.read_csv(path)
    require_columns(frame, columns, path)
    return frame


def save_figure(
    figure: Figure,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
    figure_key: str,
    title: str,
    source_files: Sequence[Path],
) -> list[dict[str, object]]:
    """统一保存 PNG/SVG，并返回清单记录。"""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for fmt in formats:
        output = output_stem.with_suffix(f".{fmt}")
        figure.savefig(
            output,
            dpi=dpi if fmt == "png" else None,
            bbox_inches="tight",
            facecolor="white",
            format=fmt,
        )
        LOGGER.info("已保存图表：%s", output)
        rows.append(
            {
                "figure_key": figure_key,
                "title": title,
                "format": fmt,
                "dpi": dpi if fmt == "png" else "vector",
                "path": str(output.resolve()),
                "source_files": " | ".join(str(path.resolve()) for path in source_files),
            }
        )
    plt.close(figure)
    return rows


def _add_box(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#284B63",
    fontsize: float = 10,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#172B3A",
        transform=ax.transAxes,
        linespacing=1.35,
    )
    return box


def _add_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#456879",
    connectionstyle: str = "arc3",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color=color,
        connectionstyle=connectionstyle,
        zorder=4,
    )
    ax.add_patch(arrow)


def draw_project_workflow() -> Figure:
    """绘制从数据到统计与解释性的项目整体流程。"""

    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.set_axis_off()
    ax.set_title("肺结节候选分类项目整体流程", fontsize=16, fontweight="bold", pad=18)
    labels = [
        "LUNA16 / LIDC-IDRI\nMHD、DICOM/XML",
        "SeriesUID 去重与划分\n训练 / 验证 / 固定测试",
        "肺窗归一化与候选裁剪\n3 × 64 × 64 patch",
        "ResNet18 特征编码\n512 维表征",
        "PBIP-Lite 原型融合\n类别原型 + Top-k 相似度",
        "冻结验证集阈值\n固定/外部测试",
        "统计与解释\nCI、检验、Grad-CAM、检索",
    ]
    colors = ["#EAF2F8", "#E8F4F0", "#FFF3E8", "#EAF2F8", "#FCEDE8", "#E8F4F0", "#F3EEF8"]
    width, height, gap = 0.118, 0.30, 0.024
    start_x = 0.008
    y = 0.36
    for index, (label, color) in enumerate(zip(labels, colors)):
        x = start_x + index * (width + gap)
        _add_box(ax, (x, y), width, height, label, color, fontsize=9.5)
        if index < len(labels) - 1:
            _add_arrow(ax, (x + width, y + height / 2), (x + width + gap, y + height / 2))
    ax.text(
        0.5,
        0.12,
        "原则：按 CT 级别防止数据泄漏；模型选择与阈值仅使用验证集；所有图表从真实日志/CSV 读取",
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
        transform=ax.transAxes,
    )
    return fig


def draw_model_architecture() -> Figure:
    """绘制与源码一致的 ResNet18 + PBIP-Lite 双分支结构。"""

    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.set_axis_off()
    ax.set_title("ResNet18 + PBIP-Lite 模型结构", fontsize=16, fontweight="bold", pad=18)

    _add_box(ax, (0.03, 0.42), 0.12, 0.18, "输入 patch\n3 × 64 × 64", "#EAF2F8")
    _add_box(
        ax,
        (0.20, 0.40),
        0.18,
        0.22,
        "ResNet18 主干\n3×3 Conv，移除 MaxPool\nGlobal Average Pooling",
        "#DCEAF5",
    )
    _add_box(ax, (0.43, 0.42), 0.13, 0.18, "特征向量 f\n512 维", "#E8F4F0")
    _add_arrow(ax, (0.15, 0.51), (0.20, 0.51))
    _add_arrow(ax, (0.38, 0.51), (0.43, 0.51))

    _add_box(ax, (0.62, 0.68), 0.15, 0.17, "分类器分支\nDropout + Linear\n基础 logit", "#EAF2F8")
    _add_box(
        ax,
        (0.60, 0.17),
        0.19,
        0.26,
        "原型分支\n类别原型库（K=3/类）\n余弦相似度 + 类内 Top-k\n正/负原型证据",
        "#FCEDE8",
    )
    _add_arrow(ax, (0.56, 0.53), (0.62, 0.76), connectionstyle="arc3,rad=-0.12")
    _add_arrow(ax, (0.56, 0.49), (0.60, 0.30), connectionstyle="arc3,rad=0.12")

    _add_box(
        ax,
        (0.83, 0.40),
        0.14,
        0.22,
        "加权融合\nz = (1−α)z_cls + αz_proto\nSigmoid 输出结节概率",
        "#E8F4F0",
    )
    _add_arrow(ax, (0.77, 0.76), (0.86, 0.61), connectionstyle="arc3,rad=0.12")
    _add_arrow(ax, (0.79, 0.30), (0.86, 0.41), connectionstyle="arc3,rad=-0.12")

    ax.text(
        0.50,
        0.04,
        "训练目标：L = BCE(y, z) + β · CE(y, 类别原型证据)；PBIP-Lite 取 β=0，完整模型取 β>0",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#334155",
        transform=ax.transAxes,
    )
    return fig


def draw_main_metrics(summary_path: Path) -> Figure:
    """绘制四方法 AUC、PR-AUC、F1 均值与标准差。"""

    frame = read_csv_checked(
        summary_path,
        [
            "method",
            "auc_mean",
            "auc_std",
            "pr_auc_mean",
            "pr_auc_std",
            "f1_mean",
            "f1_std",
        ],
    ).set_index("method")
    missing_methods = [method for method in METHOD_ORDER if method not in frame.index]
    if missing_methods:
        raise ValueError(f"主结果缺少方法：{missing_methods}")

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    x = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.23
    metric_specs = [
        ("AUC", "auc_mean", "auc_std"),
        ("PR-AUC", "pr_auc_mean", "pr_auc_std"),
        ("F1", "f1_mean", "f1_std"),
    ]
    for metric_index, (label, mean_col, std_col) in enumerate(metric_specs):
        means = frame.loc[list(METHOD_ORDER), mean_col].astype(float).to_numpy()
        stds = frame.loc[list(METHOD_ORDER), std_col].astype(float).to_numpy()
        positions = x + (metric_index - 1) * width
        bars = ax.bar(
            positions,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=label,
            color=METRIC_COLORS[label],
            edgecolor="white",
            linewidth=0.7,
            error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
        )
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.006, value + 0.006),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )
    ax.set_xticks(x, [METHOD_LABELS_ZH[method] for method in METHOD_ORDER])
    ax.set_ylabel("测试集指标（均值 ± 标准差）")
    ax.set_ylim(0.86, 1.012)
    ax.set_title("四种方法在 LUNA16 固定测试集上的性能比较", pad=48)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.015), frameon=False)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def draw_k_ablation(k_path: Path) -> Figure:
    """绘制 K=1/3/5 原型簇数消融。"""

    frame = read_csv_checked(k_path, ["k", "auc", "pr_auc", "f1"]).copy()
    for column in ("k", "auc", "pr_auc", "f1"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values("k")
    if frame["k"].astype(int).tolist() != [1, 3, 5]:
        raise ValueError(f"K 消融必须包含 1/3/5，实际为 {frame['k'].tolist()}")

    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    for label, column in (("AUC", "auc"), ("PR-AUC", "pr_auc"), ("F1", "f1")):
        ax.plot(
            frame["k"],
            frame[column],
            marker="o",
            markersize=6,
            linewidth=2.0,
            color=METRIC_COLORS[label],
            label=label,
        )
        for k_value, metric_value in zip(frame["k"], frame[column]):
            ax.annotate(
                f"{metric_value:.4f}",
                (k_value, metric_value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("每类原型簇数 K")
    ax.set_ylabel("测试集指标")
    ax.set_ylim(0.93, 1.002)
    ax.set_title("原型库 K 值消融实验", pad=48)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.015))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _prepare_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, ["method", "degradation", "level", "auc", "f1"], Path("robustness"))
    prepared = frame.copy()
    for column in ("level", "auc", "f1"):
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    missing_methods = [method for method in METHOD_ORDER if method not in set(prepared["method"])]
    missing_degradations = [
        name for name in DEGRADATION_ORDER if name not in set(prepared["degradation"])
    ]
    if missing_methods or missing_degradations:
        raise ValueError(
            f"鲁棒性数据不完整：缺失方法={missing_methods}，缺失退化={missing_degradations}"
        )
    return prepared


def draw_robustness_grid(robustness_path: Path, metric: str) -> Figure:
    """按六类退化分别绘制四方法的 AUC 或 F1 曲线。"""

    if metric not in {"auc", "f1"}:
        raise ValueError("鲁棒性指标仅支持 auc/f1")
    frame = _prepare_robustness(pd.read_csv(robustness_path))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.7), sharex=True, sharey=True)
    for ax, degradation in zip(axes.flat, DEGRADATION_ORDER):
        for method in METHOD_ORDER:
            clean = frame[
                (frame["method"] == method) & (frame["degradation"] == "clean")
            ]
            degraded = frame[
                (frame["method"] == method) & (frame["degradation"] == degradation)
            ].sort_values("level")
            if len(clean) != 1 or degraded["level"].astype(int).tolist() != [1, 2, 3, 4, 5]:
                raise ValueError(f"{method}/{degradation} 的强度 0–5 数据不完整")
            levels = np.r_[0, degraded["level"].to_numpy(dtype=float)]
            values = np.r_[float(clean.iloc[0][metric]), degraded[metric].to_numpy(dtype=float)]
            ax.plot(
                levels,
                values,
                color=METHOD_COLORS[method],
                marker="o",
                markersize=3.5,
                linewidth=1.7,
                label=METHOD_LABELS_ZH[method],
            )
        ax.set_title(DEGRADATION_LABELS_ZH[degradation], fontweight="bold")
        ax.set_xticks(range(6))
        ax.spines[["top", "right"]].set_visible(False)
    metric_label = "AUC" if metric == "auc" else "F1"
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle(
        f"六类医学图像退化下的 {metric_label} 鲁棒性曲线",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.supxlabel("退化强度等级（0=干净）", y=0.012)
    fig.supylabel(metric_label, x=0.012)
    fig.tight_layout(rect=(0.025, 0.035, 1, 0.90))
    return fig


def _select_gradcam_rows(samples_path: Path, method: str) -> pd.DataFrame:
    samples = read_csv_checked(
        samples_path,
        ["method", "category", "patch_file", "probability", "class", "prediction"],
    )
    samples = samples[samples["method"] == method].copy()
    rows = []
    for category in ("TP", "FP", "FN", "TN"):
        group = samples[samples["category"] == category]
        if group.empty:
            raise ValueError(f"Grad-CAM 样本缺少 {method}/{category}")
        rows.append(group.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def draw_gradcam_grid(
    samples_path: Path,
    patches_dir: Path,
    runs_root: Path,
    method: str,
    device: torch.device,
) -> Figure:
    """从权重重新计算 TP/FP/FN/TN 代表样本的 Grad-CAM。"""

    selected = _select_gradcam_rows(samples_path, method)
    patch_arrays = []
    for filename in selected["patch_file"]:
        path = patches_dir / str(filename)
        if not path.exists():
            raise FileNotFoundError(f"Grad-CAM patch 不存在：{path}")
        patch = np.load(path)
        if patch.shape != (3, 64, 64):
            raise ValueError(f"Grad-CAM patch 形状错误：{path} -> {patch.shape}")
        patch_arrays.append(patch.astype(np.float32, copy=False))

    selected_runs = best_run_per_method(discover_main_runs(runs_root))
    if method not in selected_runs:
        raise ValueError(f"找不到 Grad-CAM 方法权重：{method}")
    spec = selected_runs[method]
    model = load_model(spec.model_type, spec.checkpoint, spec.prototype_bank).to(device)
    batch = torch.from_numpy(np.stack(patch_arrays)).float().to(device)
    try:
        with GradCAM(model, model.backbone.layer4[-1]) as gradcam:
            cams, probabilities = gradcam(batch)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fig, axes = plt.subplots(4, 2, figsize=(7.8, 13.2))
    for row_index, (patch, cam, probability) in enumerate(
        zip(patch_arrays, cams.numpy(), probabilities.numpy())
    ):
        row = selected.iloc[row_index]
        original, overlay = overlay_cam(patch, cam)
        axes[row_index, 0].imshow(np.asarray(original))
        axes[row_index, 1].imshow(np.asarray(overlay))
        for ax in axes[row_index]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        axes[row_index, 0].set_ylabel(
            f"{CATEGORY_LABELS_ZH[str(row['category'])]}\n"
            f"y={int(row['class'])}，p={float(probability):.3f}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
            fontsize=10,
        )
    axes[0, 0].set_title("中心层原图", fontweight="bold")
    axes[0, 1].set_title("Grad-CAM 叠加", fontweight="bold")
    fig.suptitle(
        f"{METHOD_LABELS_ZH.get(method, method)}：TP / FP / FN / TN 可解释性对照",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.08, 0, 1, 0.975), h_pad=0.8, w_pad=0.4)
    return fig


def _center_slice(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"病例检索 patch 不存在：{path}")
    patch = np.load(path)
    if patch.shape != (3, 64, 64):
        raise ValueError(f"病例检索 patch 形状错误：{path} -> {patch.shape}")
    return np.clip(patch[1], 0.0, 1.0)


def draw_retrieval_grid(retrieval_path: Path, patches_dir: Path) -> Figure:
    """绘制查询病例及 Top-3 相似训练病例。"""

    frame = read_csv_checked(
        retrieval_path,
        [
            "query_patch",
            "rank",
            "patch_file",
            "label",
            "cosine_similarity",
            "prediction_probability",
        ],
    ).sort_values("rank")
    if frame["rank"].astype(int).tolist() != [1, 2, 3]:
        raise ValueError("病例检索结果必须包含 Top-1/2/3")
    query_path = Path(str(frame.iloc[0]["query_patch"]))
    query_label = "1" if "class1" in query_path.stem else ("0" if "class0" in query_path.stem else "未知")
    items = [
        (
            _center_slice(query_path),
            f"查询病例\n标签={query_label}",
            "#2F6B9A",
        )
    ]
    for row in frame.itertuples():
        items.append(
            (
                _center_slice(patches_dir / str(row.patch_file)),
                f"Top-{int(row.rank)}\n标签={int(row.label)}  "
                f"相似度={float(row.cosine_similarity):.4f}\np={float(row.prediction_probability):.4f}",
                "#2A9D8F",
            )
        )

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.3))
    for ax, (image, title, border_color) in zip(axes, items):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10, linespacing=1.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                fill=False,
                linewidth=2.5,
                edgecolor=border_color,
            )
        )
    fig.suptitle(
        "PBIP 特征空间 Top-3 相似病例检索（余弦相似度）",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成肺结节项目论文图表")
    parser.add_argument(
        "--main-summary",
        type=Path,
        default=PROJECT_ROOT / "runs/summary_v3/main_results_summary.csv",
    )
    parser.add_argument(
        "--k-ablation",
        type=Path,
        default=PROJECT_ROOT / "runs/ablations/k_ablation_results.csv",
    )
    parser.add_argument(
        "--robustness",
        type=Path,
        default=PROJECT_ROOT / "runs/robustness/robustness_detailed.csv",
    )
    parser.add_argument(
        "--gradcam-samples",
        type=Path,
        default=PROJECT_ROOT / "runs/gradcam/gradcam_samples.csv",
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=PROJECT_ROOT / "runs/retrieval/retrieval_example.csv",
    )
    parser.add_argument(
        "--patches",
        type=Path,
        default=PROJECT_ROOT / "data/processed/patches",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs/experiments_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "paper_figs/task3",
    )
    parser.add_argument("--formats", default="png,svg")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--gradcam-method", default="pbip_full")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前没有可用 GPU")
    return torch.device(raw)


def generate_all_figures(args: argparse.Namespace) -> pd.DataFrame:
    """生成全部图表，并返回可审计清单。"""

    if args.dpi < 72:
        raise ValueError("dpi 不能低于 72")
    formats = parse_formats(args.formats)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    LOGGER.info("Grad-CAM 计算设备：%s", device)

    records: list[dict[str, object]] = []
    figure_specs = [
        (
            draw_project_workflow(),
            "project_workflow",
            "肺结节候选分类项目整体流程",
            [],
        ),
        (
            draw_model_architecture(),
            "pbip_lite_architecture",
            "ResNet18 + PBIP-Lite 模型结构",
            [PROJECT_ROOT / "src/pbip_train.py"],
        ),
        (
            draw_main_metrics(args.main_summary),
            "main_metrics_comparison",
            "四种方法 AUC/PR-AUC/F1 比较",
            [args.main_summary],
        ),
        (
            draw_k_ablation(args.k_ablation),
            "k_ablation_curve",
            "原型库 K 值消融",
            [args.k_ablation],
        ),
        (
            draw_robustness_grid(args.robustness, "auc"),
            "robustness_auc_by_degradation",
            "六类图像退化 AUC 曲线",
            [args.robustness],
        ),
        (
            draw_robustness_grid(args.robustness, "f1"),
            "robustness_f1_by_degradation",
            "六类图像退化 F1 曲线",
            [args.robustness],
        ),
        (
            draw_gradcam_grid(
                args.gradcam_samples,
                args.patches,
                args.runs_root,
                args.gradcam_method,
                device,
            ),
            "gradcam_tp_fp_fn_tn",
            "Grad-CAM TP/FP/FN/TN 对照",
            [args.gradcam_samples],
        ),
        (
            draw_retrieval_grid(args.retrieval, args.patches),
            "top3_case_retrieval",
            "Top-3 相似病例检索",
            [args.retrieval],
        ),
    ]
    for figure, key, title, sources in figure_specs:
        records.extend(
            save_figure(
                figure,
                output_dir / key,
                formats,
                args.dpi,
                key,
                title,
                sources,
            )
        )
    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "figure_manifest.csv", index=False, encoding="utf-8")
    LOGGER.info("共生成 %d 个图形文件", len(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    configure_academic_style()
    args = parse_args(argv)
    generate_all_figures(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # CLI 需要保留完整堆栈并返回非零状态
        LOGGER.exception("图表生成失败：%s", error)
        sys.exit(1)
