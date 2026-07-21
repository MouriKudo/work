"""四种方法在六类、五档 CT 退化下的统一鲁棒性评估。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Windows 上先加载 OpenCV，避免与 torchvision 的延迟 DLL 顺序冲突。
from degradation import DEGRADATION_NAMES, DegradationTransform, load_degradation_config

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from experiment_utils import (
    METHOD_LABELS,
    best_run_per_method,
    discover_main_runs,
    load_yaml,
    make_loader,
)
from metrics import binary_metrics, collect_predictions, find_best_f1_threshold, load_model


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def draw_method_curves(summary: pd.DataFrame, metric: str, output: Path) -> None:
    """绘制四种方法的平均退化强度-指标曲线。"""
    width, height = 1000, 620
    left, right, top, bottom = 90, 50, 75, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 25), f"Degradation intensity vs {metric.upper()}", fill="black", font=font)
    y_min = max(0.0, float(summary[metric].min()) - 0.05)
    y_max = min(1.0, float(summary[metric].max()) + 0.03)
    if y_max <= y_min:
        y_min, y_max = 0.0, 1.0
    for fraction in np.linspace(0, 1, 6):
        value = y_min + fraction * (y_max - y_min)
        y = top + plot_h - int(fraction * plot_h)
        draw.line((left, y, left + plot_w, y), fill="#e5e7eb")
        draw.text((35, y - 6), f"{value:.2f}", fill="#374151", font=font)
    colors = ["#2563eb", "#f97316", "#16a34a", "#9333ea"]
    for method_index, (method, frame) in enumerate(summary.groupby("method", sort=True)):
        frame = frame.sort_values("level")
        points = []
        for row in frame.itertuples():
            x = left + int(int(row.level) / 5 * plot_w)
            y = top + plot_h - int((float(getattr(row, metric)) - y_min) / (y_max - y_min) * plot_h)
            points.append((x, y))
        color = colors[method_index % len(colors)]
        draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        legend_y = 48
        legend_x = left + method_index * 205
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 12), fill=color)
        draw.text(
            (legend_x + 23, legend_y), METHOD_LABELS.get(method, method), fill="black", font=font
        )
    for level in range(0, 6):
        x = left + int(level / 5 * plot_w)
        draw.text((x - 4, top + plot_h + 18), str(level), fill="black", font=font)
    draw.text((left + plot_w // 2 - 45, height - 30), "severity level", fill="black", font=font)
    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估四种方法的 CT 退化鲁棒性")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "src/configs/robustness.yaml")
    parser.add_argument("--only-method", choices=list(METHOD_LABELS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    degradation_config = load_degradation_config(project_path(config["degradation_config"]))
    metadata = project_path(config["metadata"])
    patches = project_path(config["patches"])
    output_dir = project_path(config["output_dir"])
    figure_dir = project_path(config["figure_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = best_run_per_method(discover_main_runs(project_path(config["runs_root"])))
    methods = [args.only_method] if args.only_method else config["methods"]
    rows: list[dict] = []
    selection_rows = []
    for method in methods:
        spec = selected[method]
        print(f"Evaluating {method}: seed={spec.seed}, val_auc={spec.best_val_auc:.5f}", flush=True)
        model = load_model(spec.model_type, spec.checkpoint, spec.prototype_bank).to(device)
        _, val_loader = make_loader(
            "val", metadata, patches, int(config["batch_size"]), int(config["num_workers"]), device
        )
        val_prob, val_labels, _ = collect_predictions(model, val_loader, device)
        threshold, val_f1 = find_best_f1_threshold(val_labels, val_prob)
        selection_rows.append(
            {
                "method": method,
                "seed": spec.seed,
                "best_val_auc": spec.best_val_auc,
                "threshold": threshold,
                "validation_f1": val_f1,
                "run_dir": str(spec.run_dir),
            }
        )

        _, clean_loader = make_loader(
            "test", metadata, patches, int(config["batch_size"]), int(config["num_workers"]), device
        )
        probability, labels, _ = collect_predictions(model, clean_loader, device)
        clean_metrics = binary_metrics(labels, probability, threshold)
        rows.append(
            {
                "method": method,
                "seed": spec.seed,
                "degradation": "clean",
                "level": 0,
                "auc": clean_metrics["auc"],
                "pr_auc": clean_metrics["average_precision"],
                "f1": clean_metrics["f1"],
                "threshold": threshold,
            }
        )
        for degradation in DEGRADATION_NAMES:
            for level in config["levels"]:
                transform = DegradationTransform(
                    degradation, int(level), degradation_config,
                    seed=int(config.get("seed", 0)),
                )
                _, loader = make_loader(
                    "test", metadata, patches, int(config["batch_size"]),
                    int(config["num_workers"]), device, transform,
                )
                probability, labels, _ = collect_predictions(model, loader, device)
                metrics = binary_metrics(labels, probability, threshold)
                rows.append(
                    {
                        "method": method,
                        "seed": spec.seed,
                        "degradation": degradation,
                        "level": int(level),
                        "auc": metrics["auc"],
                        "pr_auc": metrics["average_precision"],
                        "f1": metrics["f1"],
                        "threshold": threshold,
                    }
                )
                print(f"  {degradation} level={level} AUC={metrics['auc']:.4f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    detailed = pd.DataFrame(rows)
    detailed.to_csv(output_dir / "robustness_detailed.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output_dir / "selected_models.csv", index=False)
    degraded = detailed[detailed["level"] > 0]
    summary = (
        degraded.groupby(["method", "level"], as_index=False)
        .agg(auc=("auc", "mean"), pr_auc=("pr_auc", "mean"), f1=("f1", "mean"))
    )
    clean = detailed[detailed["level"] == 0][["method", "auc", "pr_auc", "f1"]].copy()
    clean["level"] = 0
    summary = pd.concat([clean, summary], ignore_index=True).sort_values(["method", "level"])
    for metric in ("auc", "pr_auc", "f1"):
        clean_map = summary[summary["level"] == 0].set_index("method")[metric]
        summary[f"{metric}_drop"] = summary.apply(
            lambda row: float(clean_map[row["method"]] - row[metric]), axis=1
        )
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)
    draw_method_curves(summary, "auc", figure_dir / "robustness_auc_curves.png")
    draw_method_curves(summary, "f1", figure_dir / "robustness_f1_curves.png")
    (output_dir / "evaluation_protocol.json").write_text(
        json.dumps(
            {
                "model_selection": "highest validation AUC among seeds 0/1/2",
                "threshold": "selected once on clean validation and fixed for all degradations",
                "aggregation": "unweighted mean across six degradation types at each level",
                "scope": "sampled-candidate test patches",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
