"""统一重评估消融实验并生成 K/beta 折线图和结果表。"""

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

from experiment_utils import evaluate_run, json_ready, load_yaml, run_spec_from_dir
from run_ablations import resolve_path


def draw_line_chart(frame: pd.DataFrame, x_column: str, title: str, output: Path) -> None:
    """使用 Pillow 绘制 AUC、PR-AUC、F1 消融折线图。"""
    width, height = 960, 600
    left, right, top, bottom = 90, 45, 70, 95
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min = max(0.0, float(frame[["auc", "pr_auc", "f1"]].min().min()) - 0.05)
    y_max = min(1.0, float(frame[["auc", "pr_auc", "f1"]].max().max()) + 0.03)
    if y_max <= y_min:
        y_min, y_max = 0.0, 1.0
    draw.text((left, 25), title, fill="black", font=font)
    for fraction in np.linspace(0, 1, 6):
        value = y_min + fraction * (y_max - y_min)
        y = top + plot_h - int(fraction * plot_h)
        draw.line((left, y, left + plot_w, y), fill="#e5e7eb")
        draw.text((35, y - 6), f"{value:.2f}", fill="#374151", font=font)
    xs = frame[x_column].astype(float).to_numpy()
    x_min, x_max = float(xs.min()), float(xs.max())
    colors = {"auc": "#2563eb", "pr_auc": "#f97316", "f1": "#16a34a"}
    def x_pixel(value: float) -> int:
        if x_max == x_min:
            return left + plot_w // 2
        return left + int((value - x_min) / (x_max - x_min) * plot_w)
    def y_pixel(value: float) -> int:
        return top + plot_h - int((value - y_min) / (y_max - y_min) * plot_h)
    for metric, color in colors.items():
        points = [(x_pixel(x), y_pixel(float(y))) for x, y in zip(xs, frame[metric])]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color)
    for x in xs:
        xp = x_pixel(float(x))
        draw.text((xp - 12, top + plot_h + 18), f"{x:g}", fill="black", font=font)
    draw.text((left + plot_w // 2 - 20, height - 35), x_column, fill="black", font=font)
    for index, (metric, color) in enumerate(colors.items()):
        x = left + index * 140
        draw.rectangle((x, 48, x + 18, 60), fill=color)
        draw.text((x + 24, 48), metric.upper(), fill="black", font=font)
    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def write_conclusion_template(frame: pd.DataFrame, output: Path) -> None:
    """将完整消融表转换为可编辑论文结论模板。"""
    k_frame = frame[frame["group"] == "k"].drop_duplicates("k")
    beta_frame = frame[frame["group"] == "beta"].drop_duplicates("beta")
    best_k = k_frame.loc[k_frame["auc"].idxmax()] if not k_frame.empty else None
    best_beta = beta_frame.loc[beta_frame["auc"].idxmax()] if not beta_frame.empty else None
    lines = [
        "# 消融实验结论模板",
        "",
        "所有 F1 阈值仅在干净验证集选择；以下结果来自 seed 0，不能替代多 seed 主表。",
        "",
    ]
    if best_k is not None:
        lines.append(
            f"- K 值实验中，测试 AUC 最佳为 K={int(best_k['k'])} "
            f"（AUC={best_k['auc']:.5f}，PR-AUC={best_k['pr_auc']:.5f}，F1={best_k['f1']:.5f}）。"
        )
    if best_beta is not None:
        lines.append(
            f"- beta 实验中，测试 AUC 最佳为 beta={best_beta['beta']:g} "
            f"（AUC={best_beta['auc']:.5f}，PR-AUC={best_beta['pr_auc']:.5f}，F1={best_beta['f1']:.5f}）。"
        )
    for row in frame[frame["group"] == "component"].itertuples():
        lines.append(
            f"- {row.method}：AUC={row.auc:.5f}，PR-AUC={row.pr_auc:.5f}，F1={row.f1:.5f}。"
        )
    lines.extend(
        [
            "- 需要结合验证曲线和跨 seed 结果讨论稳定性，不能仅按测试集最优值反向选择配置。",
            "- 当前结论属于 sampled-candidate 分类，不是官方 LUNA16 全候选 FROC。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="解析并统一评估消融实验")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "src/configs/ablation.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/ablations/ablation_results.csv")
    parser.add_argument("--figure_dir", type=Path, default=PROJECT_ROOT / "paper_figs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    root = resolve_path(args.config.resolve().parent, config.get("project_root", "../.."))
    output_root = resolve_path(root, config["output_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for experiment in config["experiments"]:
        run_dir = resolve_path(root, experiment.get("reuse_run", output_root / experiment["name"]))
        if not (run_dir / "results.json").exists():
            print(f"[missing] {experiment['name']}: {run_dir}")
            continue
        spec = run_spec_from_dir(experiment["name"], int(config["defaults"]["seed"]), run_dir)
        record = json_ready(
            evaluate_run(
                spec,
                resolve_path(root, config["metadata"]),
                resolve_path(root, config["patches"]),
                args.batch_size,
                args.num_workers,
                device=device,
            )
        )
        record.update(
            {
                "group": experiment["group"],
                "k": int(experiment["bank"].removeprefix("k")),
                "alpha": experiment.get("alpha", config["defaults"]["alpha"]),
                "beta": experiment.get("beta", config["defaults"]["beta"]),
                "reused": bool(experiment.get("reuse_run")),
            }
        )
        rows.append(record)
        print(f"evaluated {experiment['name']}")
    if not rows:
        raise RuntimeError("no completed ablation runs found")
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    k_frame = frame[frame["group"] == "k"].drop_duplicates("k").sort_values("k")
    if not k_frame.empty:
        k_frame.to_csv(args.output.parent / "k_ablation_results.csv", index=False)
        draw_line_chart(k_frame, "k", "Prototype count K ablation", args.figure_dir / "k_ablation_curve.png")
    beta_frame = frame[frame["group"] == "beta"].drop_duplicates("beta").sort_values("beta")
    if not beta_frame.empty:
        beta_frame.to_csv(args.output.parent / "beta_ablation_results.csv", index=False)
        draw_line_chart(beta_frame, "beta", "Contrastive loss beta ablation", args.figure_dir / "beta_ablation_curve.png")
    manifest = {
        "threshold_policy": "selected on clean validation only",
        "scope": "sampled candidate classification",
        "completed": frame["method"].tolist(),
    }
    (args.output.parent / "ablation_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_conclusion_template(frame, args.output.parent / "ablation_conclusion_template.md")
    print(frame[["method", "group", "k", "beta", "auc", "pr_auc", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
