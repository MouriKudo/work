"""使用干净 + 多类型退化混合增强训练 Robust-PBIP。"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Windows 上需先加载 OpenCV，再加载 torchvision 的原生运行库。
from degradation import DEGRADATION_NAMES, DegradationTransform, MixedDegradationTransform, load_degradation_config

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from experiment_utils import load_yaml, make_loader
from luna16_dataset import LUNA16Dataset
from metrics import binary_metrics, collect_predictions, find_best_f1_threshold, load_model
from pbip_train import PBIPLite, evaluate, save_training_plot, set_seed, train_one_epoch
from train import get_augmentation


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_train_transform(config: dict, degradation_config: dict):
    """组合随机退化与已有空间/强度增强。"""
    robust = config["robust_augmentation"]
    transforms = [
        MixedDegradationTransform(
            degradation_config,
            clean_probability=float(robust["clean_probability"]),
            names=robust["types"],
            levels=robust["levels"],
            seed=int(config["training"]["seed"]),
        )
    ]
    spatial = get_augmentation(config["training"].get("spatial_augment", "strong"))
    if spatial is not None:
        transforms.append(spatial)
    return T.Compose(transforms)


def predict_model(
    model: torch.nn.Module,
    split: str,
    metadata: Path,
    patches: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    transform=None,
):
    _, loader = make_loader(
        split, metadata, patches, batch_size, num_workers, device, transform
    )
    return collect_predictions(model, loader, device)


def compare_clean_degraded(
    robust_model: PBIPLite,
    normal_model: PBIPLite,
    config: dict,
    degradation_config: dict,
    device: torch.device,
) -> pd.DataFrame:
    """两个模型均在干净验证集选阈值，再评估干净和统一档位退化测试集。"""
    metadata = project_path(config["metadata"])
    patches = project_path(config["patches"])
    comparison = config["comparison"]
    batch_size = int(comparison["batch_size"])
    num_workers = int(comparison["num_workers"])
    level = int(comparison["degradation_level"])
    rows = []
    for model_name, model in (("PBIP", normal_model), ("Robust-PBIP", robust_model)):
        val_prob, val_labels, _ = predict_model(
            model, "val", metadata, patches, batch_size, num_workers, device
        )
        threshold, validation_f1 = find_best_f1_threshold(val_labels, val_prob)
        clean_prob, clean_labels, _ = predict_model(
            model, "test", metadata, patches, batch_size, num_workers, device
        )
        clean_metrics = binary_metrics(clean_labels, clean_prob, threshold)
        rows.append(
            {
                "model": model_name,
                "scenario": "clean",
                "degradation": "clean",
                "level": 0,
                "auc": clean_metrics["auc"],
                "pr_auc": clean_metrics["average_precision"],
                "f1": clean_metrics["f1"],
                "threshold": threshold,
                "validation_f1": validation_f1,
            }
        )
        degradation_rows = []
        for degradation in DEGRADATION_NAMES:
            transform = DegradationTransform(
                degradation,
                level,
                degradation_config,
                seed=int(config["training"]["seed"]),
            )
            probability, labels, _ = predict_model(
                model, "test", metadata, patches, batch_size, num_workers, device, transform
            )
            metrics = binary_metrics(labels, probability, threshold)
            row = {
                "model": model_name,
                "scenario": "degraded",
                "degradation": degradation,
                "level": level,
                "auc": metrics["auc"],
                "pr_auc": metrics["average_precision"],
                "f1": metrics["f1"],
                "threshold": threshold,
                "validation_f1": validation_f1,
            }
            rows.append(row)
            degradation_rows.append(row)
        macro = pd.DataFrame(degradation_rows)[["auc", "pr_auc", "f1"]].mean()
        rows.append(
            {
                "model": model_name,
                "scenario": "degraded_macro",
                "degradation": "six_type_mean",
                "level": level,
                "auc": float(macro["auc"]),
                "pr_auc": float(macro["pr_auc"]),
                "f1": float(macro["f1"]),
                "threshold": threshold,
                "validation_f1": validation_f1,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Robust-PBIP")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "src/configs/robust_pbip.yaml")
    parser.add_argument("--epochs", type=int, help="覆盖 YAML epoch")
    parser.add_argument("--output-dir", type=Path, help="覆盖 YAML 输出目录")
    parser.add_argument("--seed", type=int, help="覆盖 YAML seed")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    training = config["training"]
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.seed is not None:
        training["seed"] = args.seed
    output_dir = (args.output_dir or project_path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    degradation_config = load_degradation_config(project_path(config["degradation_config"]))
    set_seed(int(training["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")

    model = PBIPLite(
        project_path(config["prototype_bank"]),
        alpha=float(training["alpha"]),
        top_k=int(training["top_k"]),
        prototype_temperature=float(training["prototype_temperature"]),
        dropout=float(training["dropout"]),
    )
    best_path = output_dir / "best_model.pth"
    serializable_config = {
        **training,
        "prototype_bank": str(project_path(config["prototype_bank"])),
        "init_checkpoint": str(project_path(config["init_checkpoint"])),
        "output_dir": str(output_dir),
        "degradation_config": str(project_path(config["degradation_config"])),
        "robust_augmentation": config["robust_augmentation"],
        "method": "robust_pbip",
    }
    if not args.evaluate_only:
        init_epoch = model.load_baseline_checkpoint(project_path(config["init_checkpoint"]))
        model.to(device)
        print(f"Device={device}; initialized from baseline epoch {init_epoch}", flush=True)
        train_ds = LUNA16Dataset(
            project_path(config["metadata"]),
            project_path(config["patches"]),
            split="train",
            transform=build_train_transform(config, degradation_config),
        )
        val_ds = LUNA16Dataset(
            project_path(config["metadata"]), project_path(config["patches"]), split="val"
        )
        loader_kwargs = {
            "batch_size": int(training["batch_size"]),
            "num_workers": int(training["num_workers"]),
            "pin_memory": device.type == "cuda",
        }
        generator = torch.Generator().manual_seed(int(training["seed"]))
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            drop_last=True,
            generator=generator,
            persistent_workers=int(training["num_workers"]) > 0,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            batch_size=int(training["batch_size"]),
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=float(training["lr"]),
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(training["epochs"])
        )
        scaler = GradScaler("cuda", enabled=amp_enabled)
        history = {
            key: [] for key in (
                "train_loss", "train_cls_loss", "train_proto_loss", "train_acc",
                "val_loss", "val_proto_loss", "val_acc", "val_auc", "val_pr_auc", "val_f1",
            )
        }
        best_val_auc = float("-inf")
        best_epoch = -1
        started_at = dt.datetime.now()
        for epoch in range(1, int(training["epochs"]) + 1):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device, epoch,
                float(training["beta"]), amp_enabled,
            )
            validation = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            history["train_loss"].append(train_metrics["loss"])
            history["train_cls_loss"].append(train_metrics["cls_loss"])
            history["train_proto_loss"].append(train_metrics["proto_loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["val_loss"].append(validation["loss"])
            history["val_proto_loss"].append(validation["prototype_loss"])
            history["val_acc"].append(validation["acc"])
            history["val_auc"].append(validation["auc"])
            history["val_pr_auc"].append(validation["pr_auc"])
            history["val_f1"].append(validation["f1"])
            print(
                f"Epoch {epoch:02d}/{training['epochs']} train={train_metrics['loss']:.4f} "
                f"val_auc={validation['auc']:.4f} val_f1={validation['f1']:.4f}", flush=True
            )
            if validation["auc"] > best_val_auc:
                best_val_auc = float(validation["auc"])
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_metrics": validation,
                        "args": serializable_config,
                    },
                    best_path,
                )
        elapsed = (dt.datetime.now() - started_at).total_seconds()
        pd.DataFrame(history).to_csv(output_dir / "training_curve.csv", index=False)
        save_training_plot(history, output_dir / "training_curves.png")
        (output_dir / "config.json").write_text(
            json.dumps(serializable_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "method": "robust_pbip",
                    "best_epoch": best_epoch,
                    "best_val_auc": best_val_auc,
                    "history": history,
                    "elapsed_seconds": elapsed,
                    "args": serializable_config,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if not best_path.exists():
        raise FileNotFoundError(f"Robust-PBIP checkpoint not found: {best_path}")

    robust_model = load_model("pbip", best_path, project_path(config["prototype_bank"])).to(device)
    normal_run = project_path(config["normal_pbip_run"])
    normal_config = json.loads((normal_run / "config.json").read_text(encoding="utf-8"))
    normal_bank = project_path(normal_config["prototype_bank"])
    normal_model = load_model("pbip", normal_run / "best_model.pth", normal_bank).to(device)
    comparison = compare_clean_degraded(
        robust_model, normal_model, config, degradation_config, device
    )
    comparison.to_csv(output_dir / "clean_degraded_comparison.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
