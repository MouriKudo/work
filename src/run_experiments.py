"""Run and aggregate the four-method, three-seed LUNA16 experiment matrix.

Methods
-------
1. ``resnet18_noaug``: ResNet18 without augmentation.
2. ``resnet18_strong``: ResNet18 with strong augmentation.
3. ``pbip_lite``: class-aware prototype fusion with beta=0.
4. ``pbip_contrast``: PBIP-Lite with the selected contrastive beta.

For each seed, the prototype bank is rebuilt from that seed's strong ResNet18
checkpoint.  Runs are resumable: a directory containing ``results.json`` is
considered complete and is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable).resolve()
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "experiments_v2"
METHODS = ("resnet18_noaug", "resnet18_strong", "pbip_lite", "pbip_contrast")


def parse_int_list(value: str) -> List[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def parse_float_list(value: str) -> List[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one beta is required")
    return result


def parse_method_list(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(METHODS)
    result = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(result) - set(METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown methods: {', '.join(unknown)}")
    return result


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_command(
    command: Sequence[str],
    log_path: Path,
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = command_text(command)
    print(f"\n$ {printable}")
    if dry_run:
        return
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {printable}\n\n")
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {printable}")


def run_if_needed(
    command: Sequence[str],
    run_dir: Path,
    dry_run: bool,
) -> None:
    if (run_dir / "results.json").exists():
        print(f"SKIP completed run: {run_dir}")
        return
    run_command(command, run_dir / "train.log", dry_run)


def python_command(script_name: str, *arguments) -> List[str]:
    return [str(PYTHON), "-u", str(SRC_DIR / script_name), *(str(arg) for arg in arguments)]


def resnet_command(
    run_dir: Path,
    seed: int,
    augment: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> List[str]:
    return python_command(
        "train.py",
        "--epochs",
        epochs,
        "--batch_size",
        batch_size,
        "--augment",
        augment,
        "--seed",
        seed,
        "--num_workers",
        num_workers,
        "--output_dir",
        run_dir,
    )


def prototype_command(
    run_dir: Path,
    checkpoint: Path,
    seed: int,
    num_workers: int,
) -> List[str]:
    return python_command(
        "prototype_bank.py",
        "--checkpoint",
        checkpoint,
        "--output_dir",
        run_dir,
        "--seed",
        seed,
        "--num_workers",
        num_workers,
    )


def pbip_command(
    run_dir: Path,
    checkpoint: Path,
    bank_path: Path,
    seed: int,
    beta: float,
    alpha: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> List[str]:
    return python_command(
        "pbip_train.py",
        "--epochs",
        epochs,
        "--batch_size",
        batch_size,
        "--augment",
        "strong",
        "--alpha",
        alpha,
        "--beta",
        beta,
        "--seed",
        seed,
        "--num_workers",
        num_workers,
        "--prototype_bank",
        bank_path,
        "--init_checkpoint",
        checkpoint,
        "--output_dir",
        run_dir,
    )


def collect_results(output_root: Path) -> pd.DataFrame:
    rows = []
    for result_path in sorted(output_root.glob("seed_*/**/results.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = result.get("args", {})
        method = result.get("method", config.get("method", result_path.parent.name))
        # Normalize PBIP names to the four declared method labels.
        if method == "pbip_lite":
            normalized_method = "pbip_lite"
        elif method == "pbip_contrast":
            normalized_method = "pbip_contrast"
        elif method == "resnet18_none":
            normalized_method = "resnet18_noaug"
        elif method == "resnet18_strong":
            normalized_method = "resnet18_strong"
        else:
            continue
        metrics = result["test_metrics"]
        rows.append(
            {
                "method": normalized_method,
                "seed": int(config.get("seed", result_path.parts[-3].split("_")[-1])),
                "alpha": config.get("alpha", ""),
                "beta": config.get("beta", ""),
                "augment": config.get("augment", ""),
                "best_epoch": result.get("best_epoch", ""),
                "best_val_auc": result["best_val_auc"],
                "test_auc": metrics["auc"],
                "test_f1": metrics["f1"],
                "test_acc": metrics["acc"],
                "test_loss": metrics["loss"],
                "run_dir": str(result_path.parent.resolve()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_results(output_root: Path) -> pd.DataFrame:
    results = collect_results(output_root)
    if results.empty:
        print("No completed results are available to aggregate.")
        return results
    results = results.sort_values(["method", "seed"]).reset_index(drop=True)
    results.to_csv(output_root / "results.csv", index=False)
    summary = (
        results.groupby("method")
        .agg(
            seeds=("seed", "count"),
            auc_mean=("test_auc", "mean"),
            auc_std=("test_auc", "std"),
            f1_mean=("test_f1", "mean"),
            f1_std=("test_f1", "std"),
            acc_mean=("test_acc", "mean"),
            acc_std=("test_acc", "std"),
            val_auc_mean=("best_val_auc", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_root / "summary.csv", index=False)
    print("\nCurrent aggregate results:")
    print(results.to_string(index=False))
    print("\nPer-method summary:")
    print(summary.to_string(index=False))
    return results


def save_beta_comparison(
    output_root: Path,
    seed: int,
    beta_to_run_dir: Dict[float, Path],
) -> None:
    rows = []
    for beta, run_dir in beta_to_run_dir.items():
        result_path = run_dir / "results.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        history = result["history"]
        for epoch, (loss, auc, f1) in enumerate(
            zip(history["train_loss"], history["val_auc"], history["val_f1"]),
            start=1,
        ):
            rows.append(
                {
                    "seed": seed,
                    "beta": beta,
                    "epoch": epoch,
                    "train_loss": loss,
                    "val_auc": auc,
                    "val_f1": f1,
                }
            )
    if not rows:
        return
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_root / "beta_comparison.csv", index=False)
    from plot_utils import render_line_chart
    from PIL import Image

    auc_series = {}
    f1_series = {}
    loss_series = {}
    for beta, group in comparison.groupby("beta"):
        label = f"beta={beta:g}"
        auc_series[label] = (group["epoch"], group["val_auc"])
        f1_series[label] = (group["epoch"], group["val_f1"])
        loss_series[label] = (group["epoch"], group["train_loss"])
    charts = [
        render_line_chart(loss_series, "Training stability", "Epoch", "Total loss", width=620, height=420),
        render_line_chart(auc_series, "Validation AUC", "Epoch", "AUC", ylim=(0, 1), width=620, height=420),
        render_line_chart(f1_series, "Validation F1", "Epoch", "F1", ylim=(0, 1), width=620, height=420),
    ]
    dashboard = Image.new("RGB", (1860, 420), "white")
    for index, chart in enumerate(charts):
        dashboard.paste(chart, (620 * index, 0))
    dashboard.save(output_root / "beta_comparison.png")


def evaluate_best_model(output_root: Path, results: pd.DataFrame, dry_run: bool) -> None:
    if results.empty:
        return
    # Model selection uses validation AUC only; test scores are not consulted.
    best_row = results.loc[results["best_val_auc"].idxmax()]
    run_dir = Path(best_row["run_dir"])
    model_type = "pbip" if best_row["method"].startswith("pbip") else "resnet18"
    command = python_command(
        "metrics.py",
        "--model",
        model_type,
        "--checkpoint",
        run_dir / "best_model.pth",
        "--output_dir",
        output_root / "best_evaluation",
        "--num_workers",
        0,
    )
    if model_type == "pbip":
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        command.extend(["--prototype_bank", config["prototype_bank"]])
    run_command(command, output_root / "best_evaluation" / "evaluation.log", dry_run)
    if not dry_run:
        selection = {
            "selection_metric": "best_val_auc",
            "method": best_row["method"],
            "seed": int(best_row["seed"]),
            "best_val_auc": float(best_row["best_val_auc"]),
            "run_dir": str(run_dir),
        }
        (output_root / "best_evaluation" / "model_selection.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LUNA16 experiment matrix")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("0,1,2"))
    parser.add_argument("--methods", type=parse_method_list, default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--contrast_beta", type=float, default=0.05)
    parser.add_argument("--beta_sweep", type=parse_float_list, default=parse_float_list("0.05,0.1"))
    parser.add_argument("--sweep_seed", type=int, default=0)
    parser.add_argument("--skip_beta_sweep", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seeds": args.seeds,
        "methods": args.methods,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "alpha": args.alpha,
        "contrast_beta": args.contrast_beta,
        "beta_sweep": args.beta_sweep,
        "method_definitions": {
            "resnet18_noaug": "ResNet18, no augmentation",
            "resnet18_strong": "ResNet18, strong augmentation",
            "pbip_lite": "class-aware prototype fusion, beta=0",
            "pbip_contrast": f"PBIP-Lite plus contrastive loss, beta={args.contrast_beta}",
        },
    }
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if not args.aggregate_only:
        for seed in args.seeds:
            seed_root = output_root / f"seed_{seed}"
            noaug_dir = seed_root / "resnet18_noaug"
            strong_dir = seed_root / "resnet18_strong"
            bank_dir = seed_root / "prototype_bank"
            pbip_dir = seed_root / "pbip_lite"
            contrast_dir = seed_root / "pbip_contrast"

            if "resnet18_noaug" in args.methods:
                run_if_needed(
                    resnet_command(
                        noaug_dir,
                        seed,
                        "none",
                        args.epochs,
                        args.batch_size,
                        args.num_workers,
                    ),
                    noaug_dir,
                    args.dry_run,
                )

            needs_prototypes = any(
                method in args.methods for method in ("pbip_lite", "pbip_contrast")
            )
            if "resnet18_strong" in args.methods or needs_prototypes:
                run_if_needed(
                    resnet_command(
                        strong_dir,
                        seed,
                        "strong",
                        args.epochs,
                        args.batch_size,
                        args.num_workers,
                    ),
                    strong_dir,
                    args.dry_run,
                )

            strong_checkpoint = strong_dir / "best_model.pth"
            bank_path = bank_dir / "prototype_bank.pkl"
            if needs_prototypes and not bank_path.exists():
                run_command(
                    prototype_command(
                        bank_dir, strong_checkpoint, seed, args.num_workers
                    ),
                    bank_dir / "build.log",
                    args.dry_run,
                )

            if "pbip_lite" in args.methods:
                run_if_needed(
                    pbip_command(
                        pbip_dir,
                        strong_checkpoint,
                        bank_path,
                        seed,
                        0.0,
                        args.alpha,
                        args.epochs,
                        args.batch_size,
                        args.num_workers,
                    ),
                    pbip_dir,
                    args.dry_run,
                )
            if "pbip_contrast" in args.methods:
                run_if_needed(
                    pbip_command(
                        contrast_dir,
                        strong_checkpoint,
                        bank_path,
                        seed,
                        args.contrast_beta,
                        args.alpha,
                        args.epochs,
                        args.batch_size,
                        args.num_workers,
                    ),
                    contrast_dir,
                    args.dry_run,
                )

        if not args.skip_beta_sweep:
            sweep_seed = args.sweep_seed
            seed_root = output_root / f"seed_{sweep_seed}"
            strong_checkpoint = seed_root / "resnet18_strong" / "best_model.pth"
            bank_path = seed_root / "prototype_bank" / "prototype_bank.pkl"
            beta_runs = {}
            for beta in args.beta_sweep:
                if np.isclose(beta, args.contrast_beta):
                    run_dir = seed_root / "pbip_contrast"
                else:
                    run_dir = output_root / "beta_sweep" / f"seed_{sweep_seed}" / f"beta_{beta:g}"
                    run_if_needed(
                        pbip_command(
                            run_dir,
                            strong_checkpoint,
                            bank_path,
                            sweep_seed,
                            beta,
                            args.alpha,
                            args.epochs,
                            args.batch_size,
                            args.num_workers,
                        ),
                        run_dir,
                        args.dry_run,
                    )
                beta_runs[beta] = run_dir
            if not args.dry_run:
                save_beta_comparison(output_root, sweep_seed, beta_runs)

    if args.dry_run:
        print("\nDry run complete; no training commands were executed.")
        return
    results = aggregate_results(output_root)
    expected = len(args.seeds) * len(args.methods)
    if len(results) != expected:
        print(
            f"WARNING: results.csv has {len(results)} rows; expected {expected} "
            "for the requested experiment matrix."
        )
    else:
        evaluate_best_model(output_root, results, dry_run=False)


if __name__ == "__main__":
    main()
