"""Unified threshold selection and candidate-level evaluation for LUNA16.

The processed dataset contains sampled candidates (all positives and a 1:3
sample of negatives).  FROC values produced here are therefore explicitly
reported as *sampled-candidate FROC*, not the official LUNA16 full-candidate
FROC score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from luna16_dataset import LUNA16Dataset
from plot_utils import save_line_chart


FROC_FP_RATES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _as_numpy(values: Iterable) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("expected a one-dimensional array")
    return array


def binary_metrics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
) -> Dict[str, object]:
    """Compute binary classification metrics at a fixed threshold."""
    y_true = _as_numpy(labels).astype(np.int64)
    y_score = _as_numpy(probabilities).astype(np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError("labels and probabilities must have the same shape")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    y_pred = (y_score >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    specificity = tn / max(1, tn + fp)
    sensitivity = tp / max(1, tp + fn)
    return {
        "threshold": float(threshold),
        "accuracy": float((tn + tp) / max(1, cm.sum())),
        "auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": cm.tolist(),
    }


def find_best_f1_threshold(
    labels: Iterable[int], probabilities: Iterable[float]
) -> Tuple[float, float]:
    """Select the exact validation threshold that maximizes F1."""
    y_true = _as_numpy(labels).astype(np.int64)
    y_score = _as_numpy(probabilities).astype(np.float64)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best_value = np.nanmax(f1_values)
    tied = np.flatnonzero(np.isclose(f1_values, best_value, rtol=0, atol=1e-12))
    # Prefer the equally-good threshold closest to 0.5 for a stable tie-break.
    best_index = tied[np.argmin(np.abs(thresholds[tied] - 0.5))]
    return float(thresholds[best_index]), float(best_value)


def candidate_froc(
    labels: Iterable[int],
    probabilities: Iterable[float],
    seriesuids: Iterable[str],
    operating_fp_rates: Sequence[float] = FROC_FP_RATES,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Compute sampled-candidate FROC and sensitivity operating points."""
    y_true = _as_numpy(labels).astype(np.int64)
    y_score = _as_numpy(probabilities).astype(np.float64)
    uids = _as_numpy(seriesuids).astype(str)
    if not (y_true.shape == y_score.shape == uids.shape):
        raise ValueError("labels, probabilities and seriesuids must align")
    n_scans = len(np.unique(uids))
    if n_scans == 0:
        raise ValueError("seriesuids contains no scans")

    thresholds = np.r_[np.inf, np.sort(np.unique(y_score))[::-1], -np.inf]
    rows = []
    positive_count = max(1, int((y_true == 1).sum()))
    for threshold in thresholds:
        predicted = y_score >= threshold
        false_positives = int(np.sum(predicted & (y_true == 0)))
        true_positives = int(np.sum(predicted & (y_true == 1)))
        rows.append(
            {
                "threshold": float(threshold),
                "false_positives": false_positives,
                "fp_per_scan": false_positives / n_scans,
                "true_positives": true_positives,
                "sensitivity": true_positives / positive_count,
            }
        )
    curve = pd.DataFrame(rows).sort_values(
        ["fp_per_scan", "sensitivity"], ascending=[True, True]
    )

    # For duplicate FP rates, retain the highest attainable sensitivity and
    # enforce the monotonic FROC envelope before interpolation.
    envelope = curve.groupby("fp_per_scan", as_index=False)["sensitivity"].max()
    envelope["sensitivity"] = np.maximum.accumulate(envelope["sensitivity"])
    target_rates = np.asarray(operating_fp_rates, dtype=np.float64)
    sensitivities = np.interp(
        target_rates,
        envelope["fp_per_scan"].to_numpy(),
        envelope["sensitivity"].to_numpy(),
        left=0.0,
        right=float(envelope["sensitivity"].iloc[-1]),
    )
    operating_points = pd.DataFrame(
        {"fp_per_scan": target_rates, "sensitivity": sensitivities}
    )
    cpm = float(sensitivities.mean())
    return curve.reset_index(drop=True), operating_points, cpm


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    labels = []
    seriesuids = []
    for x, y, uids in loader:
        output = model(x.to(device, non_blocking=True))
        logits = output[0] if isinstance(output, (tuple, list)) else output
        probabilities.append(torch.sigmoid(logits).cpu())
        labels.append(y.cpu())
        seriesuids.extend(uids)
    return (
        torch.cat(probabilities).numpy(),
        torch.cat(labels).numpy(),
        np.asarray(seriesuids),
    )


def load_model(
    model_type: str,
    checkpoint_path: Path,
    prototype_bank: Path | None,
) -> torch.nn.Module:
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    config = checkpoint.get("args", {})
    if model_type == "resnet18":
        from train import ResNet18Binary

        model = ResNet18Binary(
            pretrained=False, dropout=float(config.get("dropout", 0.3))
        )
    elif model_type == "pbip":
        from pbip_train import PBIPLite

        bank_path = prototype_bank or Path(config.get("prototype_bank", ""))
        if not bank_path or not bank_path.exists():
            raise FileNotFoundError(
                "a valid --prototype_bank is required to load PBIP-Lite"
            )
        model = PBIPLite(
            bank_path,
            alpha=float(config.get("alpha", 0.3)),
            top_k=int(config.get("top_k", 20)),
            prototype_temperature=float(config.get("prototype_temperature", 0.2)),
            dropout=float(config.get("dropout", 0.3)),
        )
    elif model_type == "simplecnn":
        from train_lightweight import SimpleCNN

        model = SimpleCNN(dropout=float(config.get("dropout", 0.3)))
    else:
        raise ValueError(f"unsupported model_type: {model_type}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def save_plots(
    output_dir: Path,
    labels: np.ndarray,
    probabilities: np.ndarray,
    froc_curve: pd.DataFrame,
    froc_points: pd.DataFrame,
) -> None:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    save_line_chart(
        {
            f"ROC AUC={roc_auc_score(labels, probabilities):.4f}": (fpr, tpr),
            "chance": ([0.0, 1.0], [0.0, 1.0]),
        },
        output_dir / "roc_curve.png",
        "ROC",
        "False positive rate",
        "True positive rate",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    save_line_chart(
        {
            "FROC": (froc_curve["fp_per_scan"], froc_curve["sensitivity"]),
            "CPM points": (froc_points["fp_per_scan"], froc_points["sensitivity"]),
        },
        output_dir / "froc_curve.png",
        "Sampled-candidate FROC",
        "False-positive candidates per scan",
        "Sensitivity",
        xlim=(0.0, max(8.0, float(froc_curve["fp_per_scan"].max()))),
        ylim=(0.0, 1.0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LUNA16 checkpoint")
    parser.add_argument("--model", choices=["resnet18", "pbip", "simplecnn"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototype_bank", type=Path)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/processed/metadata.csv")
    parser.add_argument("--patches", type=Path, default=PROJECT_ROOT / "data/processed/patches")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.checkpoint, args.prototype_bank).to(device)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_dataset = LUNA16Dataset(args.metadata, args.patches, split="val")
    test_dataset = LUNA16Dataset(args.metadata, args.patches, split="test")
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    val_probabilities, val_labels, _ = collect_predictions(model, val_loader, device)
    threshold, validation_f1 = find_best_f1_threshold(val_labels, val_probabilities)
    test_probabilities, test_labels, test_uids = collect_predictions(
        model, test_loader, device
    )
    tuned_metrics = binary_metrics(test_labels, test_probabilities, threshold)
    default_metrics = binary_metrics(test_labels, test_probabilities, 0.5)
    froc_curve, froc_points, cpm = candidate_froc(
        test_labels, test_probabilities, test_uids
    )

    threshold_record = {
        "selection_split": "validation",
        "objective": "maximum_f1",
        "threshold": threshold,
        "validation_f1": validation_f1,
    }
    (output_dir / "best_threshold.json").write_text(
        json.dumps(threshold_record, indent=2), encoding="utf-8"
    )
    summary = {
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "evaluation_scope": "sampled_candidate_patches",
        "warning": (
            "Negative candidates were sampled 1:3 during preprocessing; this is not "
            "the official full-candidate LUNA16 FROC."
        ),
        "validation": threshold_record,
        "test_at_selected_threshold": tuned_metrics,
        "test_at_default_threshold": default_metrics,
        "sampled_candidate_cpm": cpm,
        "test_candidates": int(len(test_labels)),
        "test_scans": int(len(np.unique(test_uids))),
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"setting": "validation_selected", **tuned_metrics, "cpm": cpm},
            {"setting": "default_0.5", **default_metrics, "cpm": cpm},
        ]
    ).drop(columns=["confusion_matrix"]).to_csv(
        output_dir / "test_metrics.csv", index=False
    )
    froc_curve.to_csv(output_dir / "froc_curve.csv", index=False)
    froc_points.to_csv(output_dir / "froc_operating_points.csv", index=False)
    fpr, tpr, thresholds = roc_curve(test_labels, test_probabilities)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        output_dir / "roc_curve.csv", index=False
    )
    pd.DataFrame(
        {
            "seriesuid": test_uids,
            "label": test_labels,
            "probability": test_probabilities,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    save_plots(output_dir, test_labels, test_probabilities, froc_curve, froc_points)

    print(
        f"Selected threshold={threshold:.6f} on validation F1={validation_f1:.4f}; "
        f"test AUC={tuned_metrics['auc']:.4f}, F1={tuned_metrics['f1']:.4f}, "
        f"sampled-candidate CPM={cpm:.4f}"
    )
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
