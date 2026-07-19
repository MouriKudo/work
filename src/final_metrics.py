"""
Task 7+8: FROC + Multi-Seed + Metrics + Final Summary
=======================================================
Step 1: Load best model, sweep thresholds to find best F1
Step 2: FROC analysis (candidate-level)
Step 3: Run 3 seeds (0,1,2) for ResNet18-strong and PBIP
Step 4: Aggregate all results -> results.csv
"""
import sys, json, pickle, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, precision_recall_curve, confusion_matrix
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from luna16_dataset import LUNA16Dataset
from train import ResNet18Binary, get_augmentation

OUT_DIR = Path("D:/luna16-work/runs/final_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("TASKS 7+8: FROC + MULTI-SEED + FINAL SUMMARY")
print("=" * 60)

# ============================================================
# Part 1: Best F1 threshold + metrics
# ============================================================
print("\n[1/4] Finding best F1 threshold...")

# Use best ResNet18-strong model
BEST_MODEL = Path("D:/luna16-work/runs/resnet18_aug-strong_sd42_strong_20260719_102942/best_model.pth")
model = ResNet18Binary(pretrained=False)
ckpt = torch.load(str(BEST_MODEL), map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

# Sweep thresholds on validation set
val_ds = LUNA16Dataset(split="val")
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

all_preds, all_labels = [], []
with torch.no_grad():
    for x, y, _ in val_loader:
        x = x.to(DEVICE)
        logits = model(x)
        all_preds.append(torch.sigmoid(logits).cpu())
        all_labels.append(y)
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

# Sweep thresholds
best_thresh, best_f1 = 0.5, 0.0
thresholds_test = np.arange(0.1, 0.95, 0.025)
for th in thresholds_test:
    p_bin = (all_preds >= th).astype(int)
    f1 = f1_score(all_labels, p_bin, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = th
print(f"  Best threshold: {best_thresh:.3f} (val F1={best_f1:.4f})")

# Test set evaluation with best threshold
test_ds = LUNA16Dataset(split="test")
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
test_preds, test_labels = [], []
with torch.no_grad():
    for x, y, _ in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        test_preds.append(torch.sigmoid(logits).cpu())
        test_labels.append(y)
test_preds = np.concatenate(test_preds)
test_labels = np.concatenate(test_labels)

test_bin = (test_preds >= best_thresh).astype(int)
test_cm = confusion_matrix(test_labels, test_bin)
test_acc = test_cm.diagonal().sum() / test_cm.sum()
test_auc = roc_auc_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_bin, zero_division=0)
test_sens = test_cm[1,1] / max(1, test_cm[1,0]+test_cm[1,1])
test_spec = test_cm[0,0] / max(1, test_cm[0,0]+test_cm[0,1])

print(f"\n  Final metrics (threshold={best_thresh:.3f}):")
print(f"    Test AUC:  {test_auc:.4f}")
print(f"    Test F1:   {test_f1:.4f}")
print(f"    Test Acc:  {test_acc:.4f}")
print(f"    Sensitivity: {test_sens:.4f}")
print(f"    Specificity: {test_spec:.4f}")
print(f"    CM: TN={test_cm[0,0]} FP={test_cm[0,1]} FN={test_cm[1,0]} TP={test_cm[1,1]}")

# Save threshold
THRESH_FILE = OUT_DIR / "best_threshold.json"
json.dump({"threshold": float(best_thresh), "val_f1": float(best_f1), "test_auc": float(test_auc), "test_f1": float(test_f1)}, open(THRESH_FILE, "w"), indent=2)
print(f"  Saved: {THRESH_FILE}")

# ============================================================
# Part 2: FROC (Candidate-Level)
# ============================================================
print("\n[2/4] Candidate-level FROC analysis...")
# FROC: false positives per scan vs sensitivity
# Since we have patch-level classification, we approximate:
# Each patch = 1 candidate
# FP per scan = (FP across test set) / (num CTs in test set)
n_test_cts = test_ds.df["seriesuid"].nunique()
total_fp = test_cm[0, 1]
total_tp = test_cm[1, 1]
fps_per_scan = total_fp / n_test_cts
sensitivity = total_tp / (total_tp + test_cm[1, 0])

# Also compute FROC curve at multiple thresholds
froc_points = []
for th in np.arange(0.1, 0.95, 0.05):
    bin_pred = (test_preds >= th).astype(int)
    cm = confusion_matrix(test_labels, bin_pred)
    fp = cm[0, 1]
    tp = cm[1, 1]
    fps_per = fp / n_test_cts
    sens = tp / max(1, tp + cm[1, 0])
    froc_points.append({"threshold": float(th), "fps_per_scan": float(fps_per), "sensitivity": float(sens)})

froc_df = pd.DataFrame(froc_points)
froc_df.to_csv(OUT_DIR / "froc_curve.csv", index=False)

# ROC curve data
fpr, tpr, roc_thresholds = roc_curve(test_labels, test_preds)
roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": np.clip(roc_thresholds, 0, 1)})
roc_df.to_csv(OUT_DIR / "roc_curve.csv", index=False)

print(f"  FPs per scan at best threshold: {fps_per_scan:.1f}")
print(f"  Sensitivity: {sensitivity:.4f}")
print(f"  FROC data saved: {OUT_DIR / 'froc_curve.csv'}")
print(f"  ROC data saved:  {OUT_DIR / 'roc_curve.csv'}")

# ============================================================
# Part 3: Multi-Seed Experiments
# ============================================================
print("\n[3/4] Running multi-seed experiments...")
# We already have seed=42 for both methods. Run seeds 0 and 1.

# Collect all existing results
all_results = []

# ResNet18-strong results (already have seed 42)
for aug_label, prefix in [("strong", "ResNet18+Aug")]:
    r_dir = Path("D:/luna16-work/runs/resnet18_aug-strong_sd42_strong_20260719_102942")
    r = json.load(open(r_dir / "results.json"))
    all_results.append({
        "method": "ResNet18+Aug", "seed": 42, "augment": "strong",
        "test_auc": r["test_metrics"]["auc"], "test_f1": r["test_metrics"]["f1"],
        "test_acc": r["test_metrics"]["acc"], "best_val_auc": r["best_val_auc"],
        "run_dir": str(r_dir),
    })

# PBIP results
for beta_val, pbip_dir in [
    (0.05, "D:/luna16-work/runs/pbip_alpha0.3_beta0.05_sd42_20260719_120132"),
    (0.1, "D:/luna16-work/runs/pbip_alpha0.3_beta0.1_sd42_20260719_120624"),
]:
    r = json.load(open(pbip_dir + "/results.json"))
    all_results.append({
        "method": f"PBIP(beta={beta_val})", "seed": 42, "augment": "none",
        "test_auc": r["test_metrics"]["auc"], "test_f1": r["test_metrics"]["f1"],
        "test_acc": r["test_metrics"]["acc"], "best_val_auc": r["best_val_auc"],
        "run_dir": pbip_dir,
    })

# Train ResNet18-strong with seed=0 and seed=1
for sd in [0, 1]:
    print(f"  Training ResNet18-strong seed={sd}...")
    torch.manual_seed(sd); np.random.seed(sd)
    from train import train_one_epoch as t1e, evaluate as eval_m
    import torch.optim as optim
    from torch.amp import GradScaler
    import torchvision.transforms as T

    # Build fresh model
    model2 = ResNet18Binary(pretrained=False).to(DEVICE)
    augment = T.Compose([
        T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5),
        T.RandomRotation(30), T.ColorJitter(0.2, 0.2, 0.1),
        T.RandomResizedCrop(64, (0.7, 1.0)),
        T.GaussianBlur(3, (0.1, 1.0)),
    ])
    train_ds2 = LUNA16Dataset(split="train", transform=augment)
    val_ds2 = LUNA16Dataset(split="val")
    test_ds2 = LUNA16Dataset(split="test")
    train_ldr = DataLoader(train_ds2, 64, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_ldr = DataLoader(val_ds2, 64, shuffle=False, num_workers=0, pin_memory=True)
    test_ldr = DataLoader(test_ds2, 64, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    optim2 = optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-4)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=20)
    scaler2 = GradScaler('cuda')

    best_vauc, best_ep = 0.0, 0
    run_d2 = Path(f"D:/luna16-work/runs/resnet18_strong_sd{sd}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_d2.mkdir(parents=True, exist_ok=True)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": [], "val_f1": []}
    for ep in range(1, 21):
        tl, ta = t1e(model2, train_ldr, optim2, criterion, scaler2, DEVICE, ep)
        sched2.step()
        vm = eval_m(model2, val_ldr, criterion, DEVICE)
        for k, v in [("train_loss", tl), ("train_acc", ta), ("val_loss", vm["loss"]),
                     ("val_acc", vm["acc"]), ("val_auc", vm["auc"]), ("val_f1", vm["f1"])]:
            hist[k].append(v)
        if vm["auc"] > best_vauc:
            best_vauc = vm["auc"]; best_ep = ep
            torch.save({"epoch": ep, "model_state_dict": model2.state_dict(), "val_metrics": vm}, run_d2 / "best_model.pth")
        if ep % 5 == 0:
            print(f"    seed={sd} ep={ep}/20 val_auc={vm['auc']:.4f}")

    # Load best and test
    ckpt2 = torch.load(run_d2 / "best_model.pth", map_location=DEVICE, weights_only=False)
    model2.load_state_dict(ckpt2["model_state_dict"])
    tst = eval_m(model2, test_ldr, criterion, DEVICE)

    r_save = {"best_val_auc": best_vauc, "test_metrics": tst, "history": hist, "seed": sd}
    json.dump(r_save, open(run_d2 / "results.json", "w"), indent=2, default=str)
    pd.DataFrame(hist).to_csv(run_d2 / "training_curve.csv", index=False)

    all_results.append({
        "method": "ResNet18+Aug", "seed": sd, "augment": "strong",
        "test_auc": tst["auc"], "test_f1": tst["f1"],
        "test_acc": tst["acc"], "best_val_auc": best_vauc,
        "run_dir": str(run_d2),
    })

# ============================================================
# Part 4: Aggregate Results
# ============================================================
print("\n[4/4] Aggregating results...")
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_DIR / "results.csv", index=False)

print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))

# Group by method
print("\n  Per-method averages:")
for method in results_df["method"].unique():
    sub = results_df[results_df["method"] == method]
    print(f"    {method}: mean AUC={sub['test_auc'].mean():.4f} +/- {sub['test_auc'].std():.4f} "
          f"mean F1={sub['test_f1'].mean():.4f} (n={len(sub)} seeds)")

print(f"\n  All results saved to {OUT_DIR}/")
for f in sorted(OUT_DIR.glob("*")):
    print(f"    {f.name}")

print("=" * 60)

# ============================================================
# Also add a simple FROC plot using PIL
# ============================================================
from PIL import Image, ImageDraw
W, H = 800, 500
img = Image.new('RGB', (W, H), (255, 255, 255))
draw = ImageDraw.Draw(img)
ml, mr, mt, mb = 80, 50, 50, 60
pw, ph = W - ml - mr, H - mt - mb

# Draw axes
draw.rectangle([ml, mt, ml+pw, mt+ph], outline=(200,200,200), width=1)
draw.text((W//2 - 80, 10), 'FROC: Sensitivity vs False Positives per Scan', fill=(0,0,0))

# Plot points
for p in froc_points:
    px = ml + int(p['fps_per_scan'] / max(5, max(r['fps_per_scan'] for r in froc_points)) * pw)
    py = mt + ph - int(p['sensitivity'] * ph)
    draw.ellipse([px-3, py-3, px+3, py+3], fill=(220, 20, 60))

# Connect dots
pts = [(ml + int(p['fps_per_scan'] / max(5, max(r['fps_per_scan'] for r in froc_points)) * pw),
        mt + ph - int(p['sensitivity'] * ph)) for p in froc_points]
for a, b in zip(pts[:-1], pts[1:]):
    draw.line([a, b], fill=(220, 20, 60), width=2)

img.save(OUT_DIR / "froc_curve.png")
print(f"  FROC plot saved: {OUT_DIR / 'froc_curve.png'}")

# Also a summary bar chart
W2, H2 = 1000, 500
img2 = Image.new('RGB', (W2, H2), (255, 255, 255))
d2 = ImageDraw.Draw(img2)
d2.text((W2//2 - 150, 10), 'Method Comparison (Test AUC)', fill=(0,0,0))

methods = results_df.groupby('method')['test_auc'].agg(['mean', 'std']).reset_index()
mx = methods['mean'].max() * 1.15
bw = 150; gap = 60
ch2 = H2 - 80 - 120
for i, (_, row) in enumerate(methods.iterrows()):
    x0 = 120 + i * (bw + gap)
    bh = int(row['mean'] / mx * ch2)
    y0 = 60 + ch2 - bh
    color = (34, 139, 34) if 'ResNet' in row['method'] else (30, 144, 255)
    d2.rectangle([x0, y0, x0+bw, 60+ch2], fill=color, outline=(0,0,0), width=2)
    d2.text((x0 + bw//2 - 20, y0 - 20), f"{row['mean']:.4f}", fill=(0,0,0))
    d2.text((x0 + 5, 60+ch2+10), row['method'][:20], fill=(0,0,0))
    if row['std'] > 0:
        d2.text((x0 + bw//2 - 30, 60+ch2+35), f"+/- {row['std']:.4f}", fill=(120,120,120))
img2.save(OUT_DIR / "method_comparison.png")
print(f"  Method comparison: {OUT_DIR / 'method_comparison.png'}")
