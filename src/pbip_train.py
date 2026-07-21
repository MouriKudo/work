"""Train PBIP-Lite on LUNA16 candidate patches.

PBIP-Lite augments the ResNet18 classifier with class-aware prototype evidence:

    base_logit = classifier(feature)
    proto_logit = score(positive prototypes) - score(negative prototypes)
    fused_logit = (1 - alpha) * base_logit + alpha * proto_logit

When ``beta > 0``, a prototype classification loss encourages each sample to
match prototypes from its own class.  ``beta=0`` is the PBIP-Lite ablation and
``beta>0`` is PBIP-Lite with prototype contrastive supervision.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from luna16_dataset import LUNA16Dataset
from plot_utils import render_line_chart
from train import ResNet18Binary, get_augmentation


DEFAULT_BANK = (
    PROJECT_ROOT / "runs/experiments_v2/seed_0/prototype_bank/prototype_bank.pkl"
)
DEFAULT_INIT_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "experiments_v2/seed_0/resnet18_strong"
    / "best_model.pth"
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prototype_class_logits(
    cosine_similarity: torch.Tensor,
    prototype_labels: torch.Tensor,
    top_k: int = 20,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Aggregate prototype similarities into ``[negative, positive]`` logits.

    The two classes are aggregated independently.  This is important: a high
    similarity to a negative prototype must be evidence against a nodule, not
    evidence for it.
    """
    if cosine_similarity.ndim != 2:
        raise ValueError("cosine_similarity must have shape [batch, prototypes]")
    if prototype_labels.ndim != 1:
        raise ValueError("prototype_labels must have shape [prototypes]")
    if cosine_similarity.shape[1] != prototype_labels.numel():
        raise ValueError("prototype count and prototype_labels length do not match")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    class_scores = []
    for class_id in (0, 1):
        class_mask = prototype_labels == class_id
        if not torch.any(class_mask):
            raise ValueError(f"prototype bank contains no class-{class_id} prototypes")
        similarities = cosine_similarity[:, class_mask]
        k = min(top_k, similarities.shape[1])
        score = similarities.topk(k, dim=1).values.mean(dim=1)
        class_scores.append(score / temperature)
    return torch.stack(class_scores, dim=1)


def prototype_contrastive_loss(
    class_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Class-aware prototype contrastive loss.

    Cross entropy over negative/positive prototype evidence is a stable,
    non-negative supervised contrastive objective.  Unlike the previous
    implementation, labels directly determine which prototype class is pulled
    closer.
    """
    return F.cross_entropy(class_logits, labels.long())


def _load_prototype_tensors(bank_path: Path) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    with bank_path.open("rb") as handle:
        bank = pickle.load(handle)

    prototype_features = []
    prototype_labels = []
    class_mapping = (("negative", 0), ("positive", 1))
    for class_name, class_id in class_mapping:
        if class_name not in bank["prototypes"]:
            raise ValueError(f"prototype bank is missing '{class_name}' prototypes")
        for cluster in bank["prototypes"][class_name]:
            indices = np.asarray(cluster["indices"], dtype=np.int64)
            features = np.asarray(bank["features"])[indices]
            prototype_features.append(torch.as_tensor(features, dtype=torch.float32))
            prototype_labels.append(
                torch.full((len(indices),), class_id, dtype=torch.long)
            )

    features_tensor = torch.cat(prototype_features, dim=0)
    labels_tensor = torch.cat(prototype_labels, dim=0)
    return features_tensor, labels_tensor, bank


class PBIPLite(nn.Module):
    """ResNet18 binary classifier with class-aware prototype fusion."""

    def __init__(
        self,
        prototype_bank_path: str | Path,
        alpha: float = 0.3,
        top_k: int = 20,
        prototype_temperature: float = 0.2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))

        prototypes, prototype_labels, bank = _load_prototype_tensors(
            Path(prototype_bank_path)
        )
        self.register_buffer("prototypes", F.normalize(prototypes, dim=1))
        self.register_buffer("prototype_labels", prototype_labels)
        self.alpha = float(alpha)
        self.top_k = int(top_k)
        self.prototype_temperature = float(prototype_temperature)
        self.prototype_bank_metadata = {
            "checkpoint": bank.get("checkpoint"),
            "k_clusters": bank.get("k_clusters"),
            "n_prototypes": bank.get("n_prototypes"),
        }

    def load_baseline_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Initialize backbone and classifier from the bank's feature model."""
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        baseline = ResNet18Binary(pretrained=False, dropout=self.classifier[0].p)
        baseline.load_state_dict(checkpoint["model_state_dict"])

        backbone_state = {
            key: value
            for key, value in baseline.backbone.state_dict().items()
            if not key.startswith("fc.")
        }
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        meaningful_missing = [key for key in missing if not key.startswith("fc.")]
        if meaningful_missing or unexpected:
            raise RuntimeError(
                f"baseline backbone mismatch: missing={meaningful_missing}, "
                f"unexpected={unexpected}"
            )
        self.classifier.load_state_dict(baseline.backbone.fc.state_dict())
        return int(checkpoint.get("epoch", -1))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        normalized_features = F.normalize(features, dim=1)
        base_logit = self.classifier(features).squeeze(-1)

        cosine_similarity = normalized_features @ self.prototypes.T
        class_logits = prototype_class_logits(
            cosine_similarity,
            self.prototype_labels,
            top_k=self.top_k,
            temperature=self.prototype_temperature,
        )
        prototype_logit = class_logits[:, 1] - class_logits[:, 0]
        fused_logit = (1.0 - self.alpha) * base_logit + self.alpha * prototype_logit
        return fused_logit, features, cosine_similarity, class_logits


@torch.no_grad()
def evaluate(
    model: PBIPLite,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    probabilities = []
    labels = []
    total_loss = 0.0
    total_proto_loss = 0.0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _, _, class_logits = model(x)
        cls_loss = criterion(logits, y.float())
        proto_loss = prototype_contrastive_loss(class_logits, y)
        total_loss += cls_loss.item() * x.size(0)
        total_proto_loss += proto_loss.item() * x.size(0)
        probabilities.append(torch.sigmoid(logits).cpu())
        labels.append(y.cpu())

    probabilities_np = torch.cat(probabilities).numpy()
    labels_np = torch.cat(labels).numpy()
    predictions = (probabilities_np >= 0.5).astype(np.int64)
    cm = confusion_matrix(labels_np, predictions, labels=[0, 1])
    return {
        "loss": total_loss / len(loader.dataset),
        "prototype_loss": total_proto_loss / len(loader.dataset),
        "acc": float(cm.diagonal().sum() / cm.sum()),
        "auc": float(roc_auc_score(labels_np, probabilities_np)),
        "pr_auc": float(average_precision_score(labels_np, probabilities_np)),
        "f1": float(f1_score(labels_np, predictions, zero_division=0)),
        "cm": cm.tolist(),
    }


def train_one_epoch(
    model: PBIPLite,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    beta: float,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "cls_loss": 0.0, "proto_loss": 0.0}
    correct = 0
    sample_count = 0

    for batch_index, (x, y, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            logits, _, _, class_logits = model(x)
            cls_loss = criterion(logits, y.float())
            proto_loss = prototype_contrastive_loss(class_logits, y)
            loss = cls_loss + beta * proto_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = x.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["cls_loss"] += cls_loss.item() * batch_size
        totals["proto_loss"] += proto_loss.item() * batch_size
        predictions = (torch.sigmoid(logits) >= 0.5).long()
        correct += (predictions == y).sum().item()
        sample_count += batch_size

        if (batch_index + 1) % 50 == 0:
            print(
                f"  Epoch {epoch:3d} | Batch {batch_index + 1:4d}/{len(loader):4d} "
                f"| total={loss.item():.4f} cls={cls_loss.item():.4f} "
                f"proto={proto_loss.item():.4f} acc={correct / sample_count:.4f}"
            )

    return {
        "loss": totals["loss"] / sample_count,
        "cls_loss": totals["cls_loss"] / sample_count,
        "proto_loss": totals["proto_loss"] / sample_count,
        "acc": correct / sample_count,
    }


def save_training_plot(history: Dict[str, Iterable[float]], output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    loss_chart = render_line_chart(
        {
            "train total": (epochs, history["train_loss"]),
            "train BCE": (epochs, history["train_cls_loss"]),
            "train prototype": (epochs, history["train_proto_loss"]),
            "validation BCE": (epochs, history["val_loss"]),
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
    metric_chart = render_line_chart(
        {
            "validation AUC": (epochs, history["val_auc"]),
            "validation PR-AUC": (epochs, history["val_pr_auc"]),
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
    dashboard.paste(metric_chart, (1240, 0))
    dashboard.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train class-aware PBIP-Lite")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--prototype_temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--augment", choices=["none", "basic", "strong"], default="strong")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/processed/metadata.csv")
    parser.add_argument("--patches", type=Path, default=PROJECT_ROOT / "data/processed/patches")
    parser.add_argument("--prototype_bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--init_checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beta < 0:
        raise ValueError("beta must be non-negative")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    print(
        f"Device={device} epochs={args.epochs} alpha={args.alpha} beta={args.beta} "
        f"augment={args.augment} seed={args.seed}"
    )

    train_transform = get_augmentation(args.augment)
    train_ds = LUNA16Dataset(args.metadata, args.patches, split="train", transform=train_transform)
    val_ds = LUNA16Dataset(args.metadata, args.patches, split="val")
    test_ds = LUNA16Dataset(args.metadata, args.patches, split="test")
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        generator=generator,
        persistent_workers=args.num_workers > 0,
        **loader_kwargs,
    )
    eval_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_ds, shuffle=False, **eval_loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **eval_loader_kwargs)

    model = PBIPLite(
        args.prototype_bank,
        alpha=args.alpha,
        top_k=args.top_k,
        prototype_temperature=args.prototype_temperature,
        dropout=args.dropout,
    )
    init_epoch = model.load_baseline_checkpoint(args.init_checkpoint)
    model.to(device)
    print(
        f"Loaded {model.prototypes.shape[0]} prototypes and initialized from "
        f"{args.init_checkpoint} (epoch={init_epoch})"
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=amp_enabled)

    method = "pbip_lite" if args.beta == 0 else "pbip_contrast"
    if args.output_dir is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = PROJECT_ROOT / "runs" / (
            f"{method}_alpha{args.alpha}_beta{args.beta}_sd{args.seed}_{timestamp}"
        )
    else:
        run_dir = args.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "prototype_bank": str(args.prototype_bank.resolve()),
            "init_checkpoint": str(args.init_checkpoint.resolve()),
            "metadata": str(args.metadata.resolve()),
            "patches": str(args.patches.resolve()),
            "output_dir": str(run_dir),
            "method": method,
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    history = {
        "train_loss": [],
        "train_cls_loss": [],
        "train_proto_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_proto_loss": [],
        "val_acc": [],
        "val_auc": [],
        "val_pr_auc": [],
        "val_f1": [],
    }
    best_val_auc = float("-inf")
    best_path = run_dir / "best_model.pth"
    started_at = dt.datetime.now()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            epoch,
            args.beta,
            amp_enabled,
        )
        validation_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_cls_loss"].append(train_metrics["cls_loss"])
        history["train_proto_loss"].append(train_metrics["proto_loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(validation_metrics["loss"])
        history["val_proto_loss"].append(validation_metrics["prototype_loss"])
        history["val_acc"].append(validation_metrics["acc"])
        history["val_auc"].append(validation_metrics["auc"])
        history["val_pr_auc"].append(validation_metrics["pr_auc"])
        history["val_f1"].append(validation_metrics["f1"])

        print(
            f"Epoch {epoch:02d}/{args.epochs}: train={train_metrics['loss']:.4f} "
            f"(BCE={train_metrics['cls_loss']:.4f}, proto={train_metrics['proto_loss']:.4f}) "
            f"val_auc={validation_metrics['auc']:.4f} "
            f"val_f1={validation_metrics['f1']:.4f}"
        )

        if validation_metrics["auc"] > best_val_auc:
            best_val_auc = float(validation_metrics["auc"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": validation_metrics,
                    "args": config,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    elapsed_seconds = (dt.datetime.now() - started_at).total_seconds()
    results = {
        "timestamp": dt.datetime.now().isoformat(),
        "method": method,
        "args": config,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_auc": best_val_auc,
        "test_metrics": test_metrics,
        "history": history,
        "elapsed_seconds": elapsed_seconds,
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(run_dir / "training_curve.csv", index=False)
    save_training_plot(history, run_dir / "training_curves.png")

    print(
        f"Test: AUC={test_metrics['auc']:.4f} F1={test_metrics['f1']:.4f} "
        f"Acc={test_metrics['acc']:.4f}"
    )
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
