"""Train the ResNet18 baseline for LUNA16 candidate classification."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from luna16_dataset import LUNA16Dataset
from plot_utils import render_line_chart


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResNet18Binary(nn.Module):
    """ResNet18 adapted to three adjacent 64x64 CT slices."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 1)
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)
        x = backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.fc(self.forward_features(x)).squeeze(-1)


def get_augmentation(level: str = "basic"):
    """Return the augmentation pipeline used in the ablation study."""
    if level == "none":
        return None
    if level == "strong":
        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(30),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.RandomResizedCrop(64, scale=(0.7, 1.0)),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ]
        )
    if level == "basic":
        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.RandomResizedCrop(64, scale=(0.85, 1.0)),
            ]
        )
    raise ValueError(f"unknown augmentation level: {level}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    probabilities = []
    labels = []
    total_loss = 0.0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y.float())
        total_loss += loss.item() * x.size(0)
        probabilities.append(torch.sigmoid(logits).cpu())
        labels.append(y.cpu())

    probability_array = torch.cat(probabilities).numpy()
    label_array = torch.cat(labels).numpy()
    predictions = (probability_array >= 0.5).astype(np.int64)
    cm = confusion_matrix(label_array, predictions, labels=[0, 1])
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": float(cm.diagonal().sum() / cm.sum()),
        "auc": float(roc_auc_score(label_array, probability_array)),
        "f1": float(f1_score(label_array, predictions, zero_division=0)),
        "cm": cm.tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    amp_enabled: bool | None = None,
):
    model.train()
    if amp_enabled is None:
        amp_enabled = device.type == "cuda"
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_index, (x, y, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        predictions = (torch.sigmoid(logits) >= 0.5).long()
        total_correct += (predictions == y).sum().item()
        total_samples += batch_size
        if (batch_index + 1) % 50 == 0:
            print(
                f"  Epoch {epoch:3d} | Batch {batch_index + 1:4d}/{len(loader):4d} "
                f"| Loss={loss.item():.4f} Acc={total_correct / total_samples:.4f}"
            )
    return total_loss / total_samples, total_correct / total_samples


def save_training_plot(history: Dict[str, Iterable[float]], output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    loss_chart = render_line_chart(
        {
            "train": (epochs, history["train_loss"]),
            "validation": (epochs, history["val_loss"]),
        },
        "Loss",
        "Epoch",
        "Loss",
        width=620,
        height=420,
    )
    accuracy_chart = render_line_chart(
        {
            "train": (epochs, history["train_acc"]),
            "validation": (epochs, history["val_acc"]),
        },
        "Accuracy",
        "Epoch",
        "Accuracy",
        ylim=(0.0, 1.0),
        width=620,
        height=420,
    )
    metrics_chart = render_line_chart(
        {
            "validation AUC": (epochs, history["val_auc"]),
            "validation F1": (epochs, history["val_f1"]),
        },
        "Validation metrics",
        "Epoch",
        "Score",
        ylim=(0.0, 1.0),
        width=620,
        height=420,
    )
    dashboard = Image.new("RGB", (1860, 420), "white")
    dashboard.paste(loss_chart, (0, 0))
    dashboard.paste(accuracy_chart, (620, 0))
    dashboard.paste(metrics_chart, (1240, 0))
    dashboard.save(output_path)


def fit_binary_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    config: dict,
    epochs: int,
    lr: float,
    amp_enabled: bool,
) -> dict:
    """Shared supervised training loop for ResNet18 and SimpleCNN."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=amp_enabled)
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "val_f1": [],
    }
    best_val_auc = float("-inf")
    best_path = run_dir / "best_model.pth"
    started_at = dt.datetime.now()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            epoch,
            amp_enabled,
        )
        validation = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(validation["loss"])
        history["val_acc"].append(validation["acc"])
        history["val_auc"].append(validation["auc"])
        history["val_f1"].append(validation["f1"])
        print(
            f"Epoch {epoch:02d}/{epochs}: train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_auc={validation['auc']:.4f} "
            f"val_f1={validation['f1']:.4f}"
        )
        if validation["auc"] > best_val_auc:
            best_val_auc = float(validation["auc"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": validation,
                    "args": config,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    results = {
        "timestamp": dt.datetime.now().isoformat(),
        "method": config["method"],
        "args": config,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_auc": best_val_auc,
        "test_metrics": test_metrics,
        "history": history,
        "elapsed_seconds": (dt.datetime.now() - started_at).total_seconds(),
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(run_dir / "training_curve.csv", index=False)
    save_training_plot(history, run_dir / "training_curves.png")
    return results


def create_loaders(args: argparse.Namespace, device: torch.device):
    train_dataset = LUNA16Dataset(
        args.metadata,
        args.patches,
        split="train",
        transform=get_augmentation(args.augment),
    )
    val_dataset = LUNA16Dataset(args.metadata, args.patches, split="val")
    test_dataset = LUNA16Dataset(args.metadata, args.patches, split="test")
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(args.seed)
    return (
        DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            generator=generator,
            **common,
        ),
        DataLoader(val_dataset, shuffle=False, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LUNA16 ResNet18 baseline")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--augment", choices=["none", "basic", "strong"], default="none")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial_name", default="")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
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
    model = ResNet18Binary(
        pretrained=args.pretrained, dropout=args.dropout
    ).to(device)

    method = f"resnet18_{args.augment}"
    if args.output_dir is None:
        trial = f"_{args.trial_name}" if args.trial_name else ""
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = PROJECT_ROOT / "runs" / f"{method}_sd{args.seed}{trial}_{timestamp}"
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
        f"pretrained={args.pretrained}"
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
