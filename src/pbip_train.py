"""
Task 5: PBIP-Lite (Prototype-Based Image Prompting - Lite)
===========================================================
核心思路：在 ResNet18 基础上加入原型相似度融合。

架构：
  ResNet18 backbone -> 512-dim feature ->
    (a) FC -> 1-dim classification logit
    (b) cosine_sim(feature, prototypes) -> proto_logit
    -> fused_logit = (1-alpha)*logit_base + alpha*proto_logit

训练：BCE loss + prototype contrastive loss (beta factor)

用法：
  python src/pbip_train.py --epochs 20 --alpha 0.3 --beta 0.05
"""
import sys, pickle, argparse, json, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from luna16_dataset import LUNA16Dataset


# ====================== PBIP Model ======================

class PBIPLite(nn.Module):
    """ResNet18 + Prototype Cosine Similarity Fusion"""
    def __init__(self, prototype_bank_path, alpha=0.3, pretrained=False):
        super().__init__()
        # Backbone (same as ResNet18Binary but return features)
        import torchvision.models as models
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()  # Remove FC, we'll use features directly

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

        # Load prototypes
        bank = pickle.load(open(prototype_bank_path, 'rb'))
        # Build prototype tensor: (N_proto, 512)
        proto_list = []
        for class_name in ['positive', 'negative']:
            for c in range(bank['k_clusters']):
                indices = bank['prototypes'][class_name][c]['indices']
                feats = bank['features'][indices]  # (N, 512)
                proto_list.append(torch.from_numpy(feats))
        self.prototypes = torch.cat(proto_list, dim=0)  # (K*2*N, 512)
        self.n_prototypes = self.prototypes.shape[0]
        print(f"  Prototypes loaded: {self.n_prototypes} prototypes ({self.prototypes.shape})")

        self.alpha = alpha  # How much to weight prototype signal vs base classifier

    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # (B, 512)
        features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)

        # Base classification
        logit_base = self.classifier(features).squeeze(-1)  # (B,)

        # Prototype similarity
        proto_device = features.device
        if self.prototypes.device != proto_device:
            self.prototypes = self.prototypes.to(proto_device)
        proto_norm = self.prototypes / (self.prototypes.norm(dim=1, keepdim=True) + 1e-8)
        cos_sim = features_norm @ proto_norm.T  # (B, N_proto)

        # Proto logit: mean cosine sim (pos prototypes give high sim -> high prob)
        # Simple approach: mean of top-K cosine similarities
        top_k = min(20, self.n_prototypes)
        top_sim, _ = torch.topk(cos_sim, top_k, dim=1)
        proto_score = top_sim.mean(dim=1)  # (B,)

        # Fuse: proto_score is "how similar to nodule prototypes"
        # Rescale proto_score to logit space
        logit_proto = proto_score * 2.0 - 1.0  # Map [0,1] cos_sim range to ~[-1,1]

        # Weighted fusion
        logit = (1 - self.alpha) * logit_base + self.alpha * logit_proto

        return logit, features, cos_sim


# ====================== Prototype Contrastive Loss ======================

def prototype_contrastive_loss(features, cos_sim, prototypes, labels, temperature=0.07):
    """
    让正样本靠近正类原型，负样本靠近负类原型。
    features: (B, 512)
    cos_sim: (B, N_proto)
    labels: (B,)  1=positive, 0=negative
    """
    batch_size = features.shape[0]
    features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
    proto_norm = prototypes / (prototypes.norm(dim=1, keepdim=True) + 1e-8)

    loss = torch.tensor(0.0, device=features.device)
    for i in range(batch_size):
        sims = features_norm[i:i+1] @ proto_norm.T  # (1, N_proto)
        # Top-5 most similar prototypes
        top_sims, top_idx = torch.topk(sims.squeeze(), min(10, prototypes.shape[0]))
        loss += -torch.log(top_sims.mean() / temperature + 1e-8)

    return loss / batch_size


# ====================== Eval ======================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device).float()
        logit, _, _ = model(x)
        loss = criterion(logit, y)
        total_loss += loss.item() * x.size(0)
        all_preds.append(torch.sigmoid(logit).cpu())
        all_labels.append(y.cpu().long())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    pred_bin = (all_preds >= 0.5).astype(int)
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": confusion_matrix(all_labels, pred_bin).diagonal().sum() / len(all_labels),
        "auc": roc_auc_score(all_labels, all_preds),
        "f1": f1_score(all_labels, pred_bin, zero_division=0),
        "cm": confusion_matrix(all_labels, pred_bin).tolist(),
    }


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch, beta=0.05):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (x, y, _) in enumerate(loader):
        x, y = x.to(device), y.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda' if x.is_cuda else 'cpu'):
            logit, features, cos_sim = model(x)
            cls_loss = criterion(logit, y)
            proto_loss = prototype_contrastive_loss(features, cos_sim, model.prototypes, y)
            loss = cls_loss + beta * proto_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logit) >= 0.5).int()
        correct += (preds == y.int()).sum().item()
        total += x.size(0)
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(loader):4d} | Loss: {loss.item():.4f} | Acc: {correct/total:.4f}")
    return total_loss / total, correct / total


# ====================== Main ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.3, help="Proto fusion weight")
    parser.add_argument("--beta", type=float, default=0.05, help="Contrastive loss weight")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | alpha={args.alpha} | beta={args.beta} | seed={args.seed}")

    # Data
    print("\nLoading data...")
    train_ds = LUNA16Dataset(split="train")
    val_ds = LUNA16Dataset(split="val")
    test_ds = LUNA16Dataset(split="test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Model
    print("\nBuilding PBIP-Lite...")
    bank_path = PROJECT_ROOT / "runs" / "prototype_bank" / "prototype_bank.pkl"
    model = PBIPLite(str(bank_path), alpha=args.alpha, pretrained=False).to(device)
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    tag = f"alpha{args.alpha}_beta{args.beta}_sd{args.seed}"
    run_dir = PROJECT_ROOT / "runs" / f"pbip_{tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Run dir: {run_dir}")

    # Train
    print(f"\nTraining PBIP-Lite (alpha={args.alpha}, beta={args.beta})...")
    t0 = datetime.datetime.now()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": [], "val_f1": []}
    best_val_auc, best_path = 0.0, None

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, args.beta)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  === Epoch {epoch:3d}/{args.epochs} === T Loss: {t_loss:.4f} T Acc: {t_acc:.4f} | "
              f"V Loss: {val_metrics['loss']:.4f} V AUC: {val_metrics['auc']:.4f} V F1: {val_metrics['f1']:.4f} | LR: {lr_now:.2e}")
        for k, v in [("train_loss", t_loss), ("train_acc", t_acc), ("val_loss", val_metrics["loss"]),
                     ("val_acc", val_metrics["acc"]), ("val_auc", val_metrics["auc"]), ("val_f1", val_metrics["f1"])]:
            history[k].append(v)
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_path = run_dir / "best_model.pth"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": val_metrics, "history": history, "args": vars(args)}, best_path)
            print(f"  >>> Best saved (val AUC={best_val_auc:.4f})")

    elapsed = datetime.datetime.now() - t0
    print(f"\nTraining done: {elapsed} | Best val AUC: {best_val_auc:.4f}")

    # Test
    print("\nFinal test...")
    if best_path:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  Test AUC: {test_metrics['auc']:.4f}  F1: {test_metrics['f1']:.4f}  Acc: {test_metrics['acc']:.4f}")
    print(f"  CM: TN={test_metrics['cm'][0][0]} FP={test_metrics['cm'][0][1]} FN={test_metrics['cm'][1][0]} TP={test_metrics['cm'][1][1]}")

    results = {"args": vars(args), "test_metrics": test_metrics, "history": history, "best_val_auc": best_val_auc,
               "timestamp": datetime.datetime.now().isoformat()}
    json.dump(results, open(run_dir / "results.json", "w"), indent=2, default=str)
    pd.DataFrame(history).to_csv(run_dir / "training_curve.csv", index=False)
    print(f"\nSaved: {run_dir}/")


if __name__ == "__main__":
    import pandas as pd
    main()
