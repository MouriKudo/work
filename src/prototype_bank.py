"""Build a class-wise prototype bank from a trained ResNet18 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from luna16_dataset import LUNA16Dataset
from train import ResNet18Binary


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "experiments_v2/seed_0/resnet18_strong"
    / "best_model.pth"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        str(checkpoint_path), map_location=device, weights_only=False
    )
    dropout = float(checkpoint.get("args", {}).get("dropout", 0.3))
    model = ResNet18Binary(pretrained=False, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, int(checkpoint.get("epoch", -1))


@torch.no_grad()
def extract_features(
    model: ResNet18Binary,
    dataset: LUNA16Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features = []
    labels = []
    seriesuids = []
    for x, y, uids in loader:
        features.append(model.forward_features(x.to(device, non_blocking=True)).cpu())
        labels.append(y)
        seriesuids.extend(uids)
    return (
        torch.cat(features).numpy(),
        torch.cat(labels).numpy(),
        np.asarray(seriesuids),
        dataset.df["patch_file"].astype(str).to_numpy(),
    )


def cosine_kmeans(
    features: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Spherical K-means using cosine similarity."""
    if len(features) < n_clusters:
        raise ValueError("n_clusters cannot exceed the number of class samples")
    rng = np.random.default_rng(seed)
    normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    centers = normalized[rng.choice(len(normalized), n_clusters, replace=False)].copy()
    assignments = np.full(len(normalized), -1, dtype=np.int64)
    for iteration in range(1, max_iter + 1):
        # Avoid NumPy's BLAS-backed ``@`` here.  The current Windows runtime
        # can raise a native DLL exception for even tiny matrix products.
        similarities = np.sum(
            normalized[:, None, :] * centers[None, :, :], axis=2
        )
        new_assignments = np.argmax(similarities, axis=1)
        if np.array_equal(assignments, new_assignments):
            return assignments, centers, iteration
        assignments = new_assignments
        for cluster_id in range(n_clusters):
            members = normalized[assignments == cluster_id]
            if len(members) == 0:
                centers[cluster_id] = normalized[rng.integers(len(normalized))]
            else:
                center = members.mean(axis=0)
                centers[cluster_id] = center / (np.linalg.norm(center) + 1e-8)
    return assignments, centers, max_iter


def construct_bank(
    features: np.ndarray,
    labels: np.ndarray,
    seriesuids: np.ndarray,
    patch_files: np.ndarray,
    checkpoint: Path,
    checkpoint_epoch: int,
    k_clusters: int,
    n_prototypes: int,
    seed: int,
) -> dict:
    prototypes: Dict[str, list] = {}
    cluster_centers: Dict[str, np.ndarray] = {}
    cluster_stats = []
    for class_name, class_id in (("negative", 0), ("positive", 1)):
        global_indices = np.flatnonzero(labels == class_id)
        class_features = features[global_indices]
        assignments, centers, iterations = cosine_kmeans(
            class_features, k_clusters, seed=seed + class_id
        )
        normalized_features = class_features / (
            np.linalg.norm(class_features, axis=1, keepdims=True) + 1e-8
        )
        class_prototypes = []
        for cluster_id in range(k_clusters):
            member_mask = assignments == cluster_id
            member_global_indices = global_indices[member_mask]
            similarities = np.sum(
                normalized_features[member_mask] * centers[cluster_id], axis=1
            )
            count = min(n_prototypes, len(similarities))
            selected_local = np.argsort(similarities)[-count:][::-1]
            selected_global = member_global_indices[selected_local]
            selected_similarities = similarities[selected_local]
            class_prototypes.append(
                {
                    "cluster_id": cluster_id,
                    "indices": selected_global.tolist(),
                    "similarities": selected_similarities.astype(float).tolist(),
                }
            )
            cluster_stats.append(
                {
                    "class": class_name,
                    "class_id": class_id,
                    "cluster_id": cluster_id,
                    "size": int(member_mask.sum()),
                    "prototypes": int(count),
                    "iterations": int(iterations),
                    "max_similarity": float(selected_similarities[0]),
                    "min_selected_similarity": float(selected_similarities[-1]),
                }
            )
        prototypes[class_name] = class_prototypes
        cluster_centers[class_name] = centers.astype(np.float32)

    return {
        "format_version": 2,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "seed": seed,
        "k_clusters": k_clusters,
        "n_prototypes": n_prototypes,
        "feature_dim": int(features.shape[1]),
        "prototypes": prototypes,
        "cluster_centers": cluster_centers,
        "features": features.astype(np.float32),
        "labels": labels.astype(np.int64),
        "seriesuids": seriesuids.tolist(),
        "patch_files": patch_files.tolist(),
        "cluster_stats": cluster_stats,
    }


def save_prototype_grid(
    bank: dict,
    patches_dir: Path,
    output_path: Path,
) -> None:
    cell_size = 64
    margin = 4
    header = 22
    n_columns = bank["n_prototypes"]
    n_rows = bank["k_clusters"] * 2
    width = margin + n_columns * (cell_size + margin)
    height = header + margin + n_rows * (cell_size + margin)
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, 5), "negative clusters followed by positive clusters", fill=0, font=font)

    row_index = 0
    for class_name in ("negative", "positive"):
        for cluster in bank["prototypes"][class_name]:
            for column, feature_index in enumerate(cluster["indices"]):
                patch_file = bank["patch_files"][feature_index]
                patch = np.load(patches_dir / patch_file)
                center_slice = np.clip(patch[1], 0.0, 1.0)
                tile = Image.fromarray((center_slice * 255).astype(np.uint8), mode="L")
                x = margin + column * (cell_size + margin)
                y = header + margin + row_index * (cell_size + margin)
                canvas.paste(tile, (x, y))
            row_index += 1
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LUNA16 prototype bank")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "runs/prototype_bank")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/processed/metadata.csv")
    parser.add_argument("--patches", type=Path, default=PROJECT_ROOT / "data/processed/patches")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--feature_file",
        type=Path,
        help="复用已有 train_features.npz，避免为不同 K 重复提取特征",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k <= 0 or args.n <= 0:
        raise ValueError("--k and --n must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading checkpoint: {args.checkpoint}")
    if args.feature_file is not None:
        checkpoint = torch.load(
            str(args.checkpoint), map_location="cpu", weights_only=False
        )
        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        # 旧版项目自产 NPZ 将字符串保存为 object；兼容读取后立即转成 Unicode。
        # --feature_file 应仅指向本项目可信的 train_features.npz。
        with np.load(args.feature_file, allow_pickle=True) as cached:
            required = {"features", "labels", "seriesuids", "patch_files"}
            missing = required - set(cached.files)
            if missing:
                raise ValueError(f"feature file is missing keys: {sorted(missing)}")
            features = np.asarray(cached["features"], dtype=np.float32)
            labels = np.asarray(cached["labels"], dtype=np.int64)
            seriesuids = np.asarray(cached["seriesuids"], dtype=str)
            patch_files = np.asarray(cached["patch_files"], dtype=str)
        if not (len(features) == len(labels) == len(seriesuids) == len(patch_files)):
            raise ValueError("cached feature arrays have inconsistent lengths")
        print(f"Reused cached features from {args.feature_file}")
    else:
        model, checkpoint_epoch = load_feature_model(args.checkpoint, device)
        dataset = LUNA16Dataset(args.metadata, args.patches, split="train")
        features, labels, seriesuids, patch_files = extract_features(
            model, dataset, device, args.batch_size, args.num_workers
        )
    print(
        f"Extracted {features.shape}; negative={(labels == 0).sum()} "
        f"positive={(labels == 1).sum()}"
    )
    bank = construct_bank(
        features,
        labels,
        seriesuids,
        patch_files,
        args.checkpoint,
        checkpoint_epoch,
        args.k,
        args.n,
        args.seed,
    )
    with (output_dir / "prototype_bank.pkl").open("wb") as handle:
        pickle.dump(bank, handle)
    np.savez_compressed(
        output_dir / "train_features.npz",
        features=features,
        labels=labels,
        seriesuids=np.asarray(seriesuids, dtype=str),
        patch_files=np.asarray(patch_files, dtype=str),
    )
    pd.DataFrame(bank["cluster_stats"]).to_csv(
        output_dir / "cluster_stats.csv", index=False
    )
    save_prototype_grid(bank, args.patches, output_dir / "prototype_grid.png")
    config = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": bank["checkpoint_sha256"],
        "checkpoint_epoch": checkpoint_epoch,
        "metadata": str(args.metadata.resolve()),
        "patches": str(args.patches.resolve()),
        "k": args.k,
        "n": args.n,
        "seed": args.seed,
        "feature_file": str(args.feature_file.resolve()) if args.feature_file else None,
        "feature_shape": list(features.shape),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Saved prototype bank and reports to {output_dir}")


if __name__ == "__main__":
    main()
