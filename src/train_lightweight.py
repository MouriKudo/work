"""Train a lightweight SimpleCNN baseline on LUNA16 candidate patches."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train import create_loaders, fit_binary_model, set_seed


class SimpleCNN(nn.Module):
    """A small convolutional baseline with roughly 24K parameters."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SimpleCNN lightweight baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--augment", choices=["none", "basic", "strong"], default="strong")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/processed/metadata.csv")
    parser.add_argument("--patches", type=Path, default=PROJECT_ROOT / "data/processed/patches")
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    train_loader, val_loader, test_loader = create_loaders(args, device)
    model = SimpleCNN(dropout=args.dropout).to(device)
    method = f"simplecnn_{args.augment}"
    if args.output_dir is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = PROJECT_ROOT / "runs" / f"{method}_sd{args.seed}_{timestamp}"
    else:
        run_dir = args.output_dir.resolve()
    config = vars(args).copy()
    config.update(
        {
            "metadata": str(args.metadata.resolve()),
            "patches": str(args.patches.resolve()),
            "output_dir": str(run_dir),
            "method": method,
        }
    )
    print(
        f"Device={device} method={method} epochs={args.epochs} seed={args.seed} "
        f"parameters={sum(p.numel() for p in model.parameters()):,}"
    )
    results = fit_binary_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        run_dir,
        config,
        args.epochs,
        args.lr,
        amp_enabled,
    )
    metrics = results["test_metrics"]
    print(
        f"Test: AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} "
        f"Acc={metrics['acc']:.4f}"
    )
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
