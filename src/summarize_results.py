"""统一汇总四种方法的 AUC、PR-AUC、F1 并生成论文图表。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_utils import METHOD_DIRS, METHOD_LABELS, discover_main_runs, evaluate_run, json_ready

METHOD_ORDER = {method: index for index, method in enumerate(METHOD_DIRS)}


def draw_grouped_bars(summary: pd.DataFrame, output_path: Path) -> None:
    """使用 Pillow 绘制三指标分组柱状图，避免环境中的 matplotlib 崩溃。"""
    width, height = 1200, 680
    left, right, top, bottom = 90, 40, 70, 130
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    plot_w, plot_h = width - left - right, height - top - bottom
    draw.text((left, 24), "Four-method test metrics (mean +/- std over seeds)", fill="black", font=font)
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h - int(tick * plot_h)
        draw.line((left, y, left + plot_w, y), fill="#e5e7eb", width=1)
        draw.text((45, y - 6), f"{tick:.1f}", fill="#4b5563", font=font)
    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)

    metrics = [("auc", "AUC", "#2563eb"), ("pr_auc", "PR-AUC", "#f97316"), ("f1", "F1", "#16a34a")]
    group_w = plot_w / len(summary)
    bar_w = min(54, group_w / 4.5)
    for group_index, row in summary.reset_index(drop=True).iterrows():
        center = left + (group_index + 0.5) * group_w
        for metric_index, (metric, _, color) in enumerate(metrics):
            value = float(row[f"{metric}_mean"])
            std = float(row[f"{metric}_std"])
            x0 = int(center + (metric_index - 1) * (bar_w + 7) - bar_w / 2)
            x1 = int(x0 + bar_w)
            y0 = top + plot_h - int(value * plot_h)
            draw.rectangle((x0, y0, x1, top + plot_h), fill=color)
            draw.text((x0 + 5, max(top, y0 - 14)), f"{value:.3f}", fill="#111827", font=font)
            error_top = top + plot_h - int(min(1.0, value + std) * plot_h)
            error_bottom = top + plot_h - int(max(0.0, value - std) * plot_h)
            error_x = (x0 + x1) // 2
            draw.line((error_x, error_top, error_x, error_bottom), fill="black", width=2)
            draw.line((error_x - 5, error_top, error_x + 5, error_top), fill="black", width=2)
            draw.line((error_x - 5, error_bottom, error_x + 5, error_bottom), fill="black", width=2)
        label = METHOD_LABELS.get(row["method"], row["method"])
        draw.text((int(center - len(label) * 3), top + plot_h + 18), label, fill="black", font=font)
    legend_x = left + 15
    for metric_index, (_, label, color) in enumerate(metrics):
        x = legend_x + metric_index * 130
        draw.rectangle((x, 50, x + 18, 62), fill=color)
        draw.text((x + 24, 50), label, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_conclusion_template(summary: pd.DataFrame, output_path: Path) -> None:
    best_auc = summary.loc[summary["auc_mean"].idxmax()]
    best_pr = summary.loc[summary["pr_auc_mean"].idxmax()]
    best_f1 = summary.loc[summary["f1_mean"].idxmax()]
    text = f"""# 主结果结论模板

## 实验设置

在相同的 LUNA16 sampled-candidate 划分上比较四种方法，并对 seed 0、1、2
报告均值与标准差。F1 阈值仅由各模型的干净验证集选择，测试集不参与调参。

## 可直接修改的结论

- 平均 AUC 最佳方法为 **{METHOD_LABELS[best_auc['method']]}**：
  {best_auc['auc_mean']:.5f} ± {best_auc['auc_std']:.5f}。
- 平均 PR-AUC 最佳方法为 **{METHOD_LABELS[best_pr['method']]}**：
  {best_pr['pr_auc_mean']:.5f} ± {best_pr['pr_auc_std']:.5f}。
- 平均 F1 最佳方法为 **{METHOD_LABELS[best_f1['method']]}**：
  {best_f1['f1_mean']:.5f} ± {best_f1['f1_std']:.5f}。
- 应同时讨论 PBIP 的平均性能与跨 seed 方差，不能只引用单个 seed。
- 本表基于 1:3 负采样后的候选分类；FROC/CPM 不能表述为官方全候选 LUNA16 结果。
"""
    output_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总四方法三 seed 正式结果")
    parser.add_argument("--runs_root", type=Path, default=PROJECT_ROOT / "runs/experiments_v2")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "runs/summary_v3")
    parser.add_argument("--figure_dir", type=Path, default=PROJECT_ROOT / "paper_figs")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/processed/metadata.csv")
    parser.add_argument("--patches", type=Path, default=PROJECT_ROOT / "data/processed/patches")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--reuse", action="store_true", help="若 main_results.csv 已存在则只重新汇总/画图")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "main_results.csv"
    if args.reuse and results_path.exists():
        results = pd.read_csv(results_path)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows = []
        for spec in discover_main_runs(args.runs_root):
            print(f"Evaluating {spec.method} seed={spec.seed}", flush=True)
            record = evaluate_run(
                spec,
                args.metadata,
                args.patches,
                args.batch_size,
                args.num_workers,
                device=device,
            )
            rows.append(json_ready(record))
        results = pd.DataFrame(rows)
    results["_order"] = results["method"].map(METHOD_ORDER)
    results = results.sort_values(["_order", "seed"]).drop(columns="_order")
    results.to_csv(results_path, index=False)

    summary = (
        results.groupby("method", as_index=False)
        .agg(
            seeds=("seed", "count"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            threshold_mean=("threshold", "mean"),
        )
    )
    summary["_order"] = summary["method"].map(METHOD_ORDER)
    summary = summary.sort_values("_order").drop(columns="_order")
    summary.to_csv(output_dir / "main_results_summary.csv", index=False)
    draw_grouped_bars(summary, args.figure_dir / "main_metrics_comparison.png")
    write_conclusion_template(summary, output_dir / "conclusion_template.md")
    (output_dir / "summary_manifest.json").write_text(
        json.dumps(
            {
                "selection": "F1 thresholds selected on validation only",
                "evaluation_scope": "sampled candidate patches",
                "rows": len(results),
                "methods": sorted(results["method"].unique().tolist()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Saved summary to {output_dir}")


if __name__ == "__main__":
    main()
