"""
Task 4: Prototype Bank Construction
====================================
1. 提取训练集特征 (ResNet18 倒数第二层 512-dim)
2. 按正负类分别 K=3 余弦聚类
3. 每聚类取 N=20 个最靠近中心的样本作为原型
4. 保存 prototype_bank.pkl + 原型样本网格图

Usage: python src/prototype_bank.py
"""
import sys, pickle, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from luna16_dataset import LUNA16Dataset

# --- Config ---
CHECKPOINT = PROJECT_ROOT / "runs" / "resnet18_aug-strong_sd42_strong_20260719_102942" / "best_model.pth"
OUT_DIR = PROJECT_ROOT / "runs" / "prototype_bank"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_CLUSTERS = 3       # 每类 k 个聚类
N_PROTOTYPES = 20     # 每聚类 N 个原型
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("TASK 4: PROTOTYPE BANK CONSTRUCTION")
print(f"  K={K_CLUSTERS}  N={N_PROTOTYPES}")
print(f"  Checkpoint: {CHECKPOINT}")
print("=" * 60)


def load_model(checkpoint_path):
    from train import ResNet18Binary
    model = ResNet18Binary(pretrained=False)
    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded model from epoch {ckpt.get('epoch', '?')}")
    return model, model.backbone.avgpool  # hook here? We'll just use forward hook


# ================================================================
# Extract 512-dim features
# ================================================================
print("\n[1/4] Extracting features from train set...")

model, _ = load_model(CHECKPOINT)

# Hook the penultimate layer (global avg pooling output => 512)
features_list = []
labels_list = []
uids_list = []
patch_names = []

def hook_fn(module, input, output):
    # output shape: (B, 512, 1, 1)
    features_list.append(output.squeeze(-1).squeeze(-1).detach().cpu())

model.backbone.avgpool.register_forward_hook(hook_fn)

ds = LUNA16Dataset(split="train")
loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

with torch.no_grad():
    for x, y, uid in loader:
        x = x.to(DEVICE)
        model(x)  # forward triggers hook
        labels_list.append(y)
        uids_list.extend(uid)

all_features = torch.cat(features_list, dim=0).numpy()  # (3553, 512)
all_labels = torch.cat(labels_list, dim=0).numpy()       # (3553,)
all_names = uids_list

print(f"  Extracted: {all_features.shape} features")
print(f"  Pos: {(all_labels==1).sum()}, Neg: {(all_labels==0).sum()}")

# ================================================================
# K=3 Cosine Similarity Clustering
# ================================================================
print(f"\n[2/4] Cosine-similarity clustering (K={K_CLUSTERS})...")

def cosine_kmeans(X, n_clusters, max_iter=100, seed=42):
    """K-means with cosine similarity (argmin of 1 - cosine_sim)"""
    np.random.seed(seed)
    idx = np.random.choice(len(X), n_clusters, replace=False)
    centers = X[idx].copy()
    # L2 normalize
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    for _ in range(max_iter):
        centers_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        sim = X_norm @ centers_norm.T  # cosine similarity
        labels = np.argmax(sim, axis=1)
        old_centers = centers.copy()
        for c in range(n_clusters):
            cluster_X = X[labels == c]
            if len(cluster_X) > 0:
                centers[c] = cluster_X.mean(axis=0)
        if np.allclose(old_centers, centers):
            break
    return labels, centers

# Cluster positive and negative separately
prototypes = {}
all_sim_results = []

for class_id, class_name in [(1, "positive"), (0, "negative")]:
    mask = all_labels == class_id
    X_class = all_features[mask]
    n_samples = len(X_class)
    print(f"\n  Class '{class_name}': {n_samples} samples")

    labels_c, centers = cosine_kmeans(X_class, n_clusters=K_CLUSTERS)

    # L2 normalize centers and features for cosine sim
    centers_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    X_norm = X_class / (np.linalg.norm(X_class, axis=1, keepdims=True) + 1e-8)
    sim = X_norm @ centers_norm.T

    cluster_protos = []
    for c in range(K_CLUSTERS):
        c_mask = labels_c == c
        c_indices = np.where(mask)[0][c_mask]
        c_sim = sim[c_mask, c]
        n_top = min(N_PROTOTYPES, len(c_sim))
        top_k = np.argsort(c_sim)[-n_top:][::-1]
        top_indices = c_indices[top_k]
        top_sims = c_sim[top_k]
        cluster_protos.append({
            "cluster_id": c,
            "indices": top_indices.tolist(),
            "similarities": top_sims.tolist(),
        })
        all_sim_results.append({
            "class": class_name,
            "cluster_id": c,
            "size": int(c_mask.sum()),
            "prototypes": n_top,
        })
        print(f"    Cluster {c}: {c_mask.sum()} samples, {n_top} prototypes (top sim={top_sims[0]:.4f})")
    prototypes[class_name] = cluster_protos

# ================================================================
# Pickle & save
# ================================================================
print(f"\n[3/4] Saving prototype bank...")

bank = {
    "k_clusters": K_CLUSTERS,
    "n_prototypes": N_PROTOTYPES,
    "feature_dim": all_features.shape[1],
    "prototypes": prototypes,
    "features": all_features,
    "labels": all_labels,
    "seriesuids": all_names,
    "cluster_stats": all_sim_results,
}

pkl_path = OUT_DIR / "prototype_bank.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(bank, f)
print(f"  Saved: {pkl_path} ({pkl_path.stat().st_size / 1024:.1f} KB)")

# Save features separately
np.savez_compressed(OUT_DIR / "train_features.npz", features=all_features, labels=all_labels)
print(f"  Saved: {OUT_DIR / 'train_features.npz'}")

# ================================================================
# Prototype sample grid
# ================================================================
print(f"\n[4/4] Generating prototype sample grid...")

PATCHES_DIR = PROJECT_ROOT / "data" / "processed" / "patches"
meta = pd.read_csv(str(PROJECT_ROOT / "data" / "processed" / "metadata.csv"))
meta['split'] = meta['split'].astype(str)
meta.loc[meta['split'].isin(['nan', '']), 'split'] = ''
meta.loc[meta['subset_id'].isin(range(0,8)), 'split'] = 'train'
meta.loc[meta['subset_id'] == 8, 'split'] = 'val'
meta.loc[meta['subset_id'] == 9, 'split'] = 'test'
train_meta = meta[meta['split'] == 'train']

total_cells = K_CLUSTERS * 2  # 3 pos + 3 neg clusters
n_cols = N_PROTOTYPES
n_rows = total_cells
cs, mg = 64, 4
gw = n_cols * (cs + mg) + mg
gh = n_rows * (cs + mg) + mg + 40
grid = np.ones((gh, gw), dtype=np.float32)

row_idx = 0
for class_name in ["positive", "negative"]:
    for c in range(K_CLUSTERS):
        proto_indices = prototypes[class_name][c]["indices"]
        for pi, idx in enumerate(proto_indices[:N_PROTOTYPES]):
            row = train_meta.iloc[idx]
            patch = np.load(PATCHES_DIR / row["patch_file"])
            ys = 20 + row_idx * (cs + mg) + mg
            xs = pi * (cs + mg) + mg
            grid[ys:ys+cs, xs:xs+cs] = patch[1]  # center slice
        row_idx += 1

cv2.imwrite(str(OUT_DIR / "prototype_grid.png"), (np.clip(grid, 0, 1) * 255).astype(np.uint8))
print(f"  Saved: {OUT_DIR / 'prototype_grid.png'}")

print(f"\nAll deliverables in {OUT_DIR}/")
print(f"  prototype_bank.pkl")
print(f"  train_features.npz")
print(f"  prototype_grid.png")
print("=" * 60)
