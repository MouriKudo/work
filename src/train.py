"""
LUNA16 ResNet18 基线训练脚本
=============================
基于 torchvision 预训练 ResNet18，微调最后一层做肺结节二分类。

用法:
  python src/train.py --epochs 3                 # 3 epoch 验证流程
  python src/train.py --epochs 20 --amp          # 20 epoch 完整训练
  python src/train.py --epochs 20 --augment      # 数据增强训练

输出:
  runs/ 目录下保存最佳模型权重、训练曲线、日志
"""
import os, sys, argparse, datetime, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision.models as models
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

import torchvision.transforms as T

# 把 src 加入 PATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from luna16_dataset import LUNA16Dataset


# ====================== 模型定义 ======================

class ResNet18Binary(nn.Module):
    """预训练 ResNet18 -> 自定义第一层 + 最后层适配 3 通道 64x64 patch"""
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # 替换第一层: 原始是 7x7 stride=2 (for 224x224), 改成 3x3 stride=1 (for 64x64)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # 去掉 maxpool，保留空间信息

        # 替换分类头
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)  # (B,) logits


# ====================== 数据增强 ======================

def get_augmentation():
    """训练集增强: 旋转 + 翻转 + 裁剪 + 强度扰动 (torchvision)"""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.RandomResizedCrop(64, scale=(0.85, 1.0)),
    ])


# ====================== 训练 / 验证 / 测试 ======================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """返回 {loss, acc, auc, f1, cm}"""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device).float()
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = torch.sigmoid(logits)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu().long())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    pred_binary = (all_preds >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, pred_binary, zero_division=0)
    cm = confusion_matrix(all_labels, pred_binary)
    acc = cm.diagonal().sum() / cm.sum()
    avg_loss = total_loss / len(loader.dataset)

    return {"loss": avg_loss, "acc": acc, "auc": auc, "f1": f1, "cm": cm.tolist()}


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch_idx, (x, y, _) in enumerate(loader):
        x, y = x.to(device), y.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda' if x.is_cuda else 'cpu'):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).int()
        total_correct += (preds == y.int()).sum().item()
        total_samples += x.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(loader):4d} | "
                  f"Loss: {loss.item():.4f} | Acc: {total_correct/total_samples:.4f}")

    return total_loss / total_samples, total_correct / total_samples


# ====================== 主函数 ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数 (3=验证, 20=基线)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--amp", action="store_true", default=True, help="混合精度 (默认开)")
    parser.add_argument("--augment", action="store_true", help="开启数据增强")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | "
          f"Augment: {args.augment} | Seed: {args.seed}")

    # --- Transforms ---
    train_aug = get_augmentation() if args.augment else None

    # --- DataLoaders ---
    print("\n" + "=" * 50)
    print("Loading data...")
    train_ds = LUNA16Dataset(split="train", transform=train_aug)
    val_ds = LUNA16Dataset(split="val")
    test_ds = LUNA16Dataset(split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # --- Model ---
    print("\n" + "=" * 50)
    print("Building model...")
    model = ResNet18Binary(pretrained=True, dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ResNet18: {total_params:,} total params | {trainable_params:,} trainable")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Run dir ---
    run_dir = PROJECT_ROOT / "runs" / f"resnet18_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Run dir: {run_dir}")

    # Log config to file immediately
    cfg_path = run_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"  Config saved to {cfg_path}")

    # --- Training ---
    print("\n" + "=" * 50)
    print("TRAINING STARTED")
    print("=" * 50)
    t0 = datetime.datetime.now()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": [], "val_f1": []}
    best_val_auc = 0.0
    best_model_path = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]['lr']

        print(f"  === Epoch {epoch:3d}/{args.epochs} === "
              f"T Loss: {train_loss:.4f} | T Acc: {train_acc:.4f} | "
              f"V Loss: {val_metrics['loss']:.4f} | V Acc: {val_metrics['acc']:.4f} | "
              f"V AUC: {val_metrics['auc']:.4f} | V F1: {val_metrics['f1']:.4f} | "
              f"LR: {lr_now:.2e}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_f1"].append(val_metrics["f1"])

        # 保存最佳模型
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            # Make sure directory still exists
            run_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = run_dir / "best_model.pth"
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": {k: v for k, v in val_metrics.items() if k != "cm"},
                "history": history,
                "args": vars(args),
            }
            torch.save(checkpoint, best_model_path)
            print(f"  >>> Best model saved (val AUC={best_val_auc:.4f})")

    elapsed = datetime.datetime.now() - t0
    print(f"\nTraining completed in {elapsed}")
    print(f"Best val AUC: {best_val_auc:.4f}")

    # --- 最终测试 ---
    print("\n" + "=" * 50)
    print("FINAL TEST EVALUATION")
    print("=" * 50)
    if best_model_path and best_model_path.exists():
        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        except Exception:
            torch.serialization.add_safe_globals([np._core.multiarray.scalar])
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Test Acc:  {test_metrics['acc']:.4f}")
    print(f"  Test AUC:  {test_metrics['auc']:.4f}")
    print(f"  Test F1:   {test_metrics['f1']:.4f}")
    print(f"  Confusion Matrix: {test_metrics['cm']}")
    print(f"    TN={test_metrics['cm'][0][0]} FP={test_metrics['cm'][0][1]}")
    print(f"    FN={test_metrics['cm'][1][0]} TP={test_metrics['cm'][1][1]}")

    # 保存结果
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "args": vars(args),
        "test_metrics": test_metrics,
        "history": history,
        "best_val_auc": best_val_auc,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 保存训练曲线数据
    pd.DataFrame(history).to_csv(run_dir / "training_curve.csv", index=False)

    print(f"\nResults saved to {run_dir}")
    print(f"  best_model.pth ({best_model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  results.json")
    print(f"  training_curve.csv")


if __name__ == "__main__":
    import pandas as pd
    main()
