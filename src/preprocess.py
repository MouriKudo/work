"""
LUNA16 预处理 Pipeline
======================
窗宽窗位归一化 + 正负样本处理 + 生成 metadata.csv

输出:
  data/processed/patches/     — .npy patches (3, 64, 64)
  data/processed/metadata.csv — 元数据表
"""

import os
import sys
import io
import re
import argparse
import hashlib
from pathlib import Path
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    import SimpleITK as sitk
except ImportError:
    print("ERROR: pip install SimpleITK")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PATCHES_DIR = PROCESSED_DIR / "patches"
FIGURES_DIR = PROJECT_ROOT / "paper_figs"


# ==================== 窗宽窗位 ====================

def lung_window(hu_array, width=1500, level=-600):
    """
    肺窗归一化: 裁剪到 [hu_min, hu_max] → 归一化到 [0, 1]
    """
    hu_min = level - width / 2.0
    hu_max = level + width / 2.0
    clipped = np.clip(hu_array, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


# ==================== CT 加载 ====================

def load_ct_scan(mhd_path):
    image = sitk.ReadImage(str(mhd_path))
    ct = sitk.GetArrayFromImage(image)  # (z, y, x)
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    seriesuid = os.path.splitext(os.path.basename(mhd_path))[0]
    return ct, origin, spacing, seriesuid


# ==================== 坐标转换 ====================

def world_to_voxel(world_xyz, origin, spacing):
    x = int(round((world_xyz[0] - origin[0]) / spacing[0]))
    y = int(round((world_xyz[1] - origin[1]) / spacing[1]))
    z = int(round((world_xyz[2] - origin[2]) / spacing[2]))
    return (x, y, z)


# ==================== Patch 提取 ====================

def extract_three_layer_patch(ct_array, voxel_xyz, patch_size=64):
    max_z, max_y, max_x = ct_array.shape
    half = patch_size // 2
    x, y, z = voxel_xyz

    x_start = max(0, x - half)
    y_start = max(0, y - half)
    x_end = min(max_x, x_start + patch_size)
    y_end = min(max_y, y_start + patch_size)
    x_start = max(0, x_end - patch_size)
    y_end = min(max_y, y_start + patch_size)
    y_start = max(0, y_end - patch_size)
    x_end = min(max_x, x_start + patch_size)
    y_end = min(max_y, y_start + patch_size)

    layers = []
    for dz in [-1, 0, 1]:
        z_idx = max(0, min(max_z - 1, z + dz))
        layer = ct_array[z_idx, y_start:y_end, x_start:x_end].astype(np.float32)
        # 确保尺寸正确
        if layer.shape != (patch_size, patch_size):
            layer_padded = np.zeros((patch_size, patch_size), dtype=np.float32)
            h, w = layer.shape
            layer_padded[:h, :w] = layer
            layer = layer_padded
        layer = lung_window(layer)
        layers.append(layer)

    return np.stack(layers, axis=0)  # (3, 64, 64)


# ==================== 正负例匹配 ====================

def match_candidate_to_annotation(candidate_coord, annotations_df, seriesuid, tolerance_mm=5.0):
    anns = annotations_df[annotations_df["seriesuid"] == seriesuid]
    if len(anns) == 0:
        return False
    cx, cy, cz = candidate_coord
    for _, row in anns.iterrows():
        dx = cx - row["coordX"]
        dy = cy - row["coordY"]
        dz = cz - row["coordZ"]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dist <= tolerance_mm:
            return True
    return False


# ==================== Subset 映射 ====================

def build_subset_map(raw_dir):
    """遍历 data/raw/ 中所有 subset 目录，建立 {seriesuid: subset_id} 映射
    支持嵌套解压结构: subset0/subset0/*.mhd 和 扁平结构: subset0/*.mhd
    """
    subset_map = {}
    for subset_dir in sorted(raw_dir.glob("subset*")):
        if not subset_dir.is_dir():
            continue
        try:
            subset_id = int(re.search(r"(\d+)", subset_dir.name).group(1))
        except (AttributeError, ValueError):
            continue

        # 查找 .mhd 文件：支持嵌套 (subset0/subset0/*.mhd) 和扁平 (subset0/*.mhd)
        mhd_inner = sorted(subset_dir.glob("*.mhd"))
        mhd_nested = sorted(subset_dir.glob("*/*.mhd"))
        mhd_files = mhd_inner if mhd_inner else mhd_nested

        for mhd_path in mhd_files:
            seriesuid = mhd_path.stem
            subset_map[seriesuid] = subset_id
    print(f"  Built subset map: {len(subset_map)} seriesuids from {len(list(raw_dir.glob('subset*')))} subset dirs")
    return subset_map


# ==================== 主 Pipeline ====================

def run_preprocessing(
    raw_dir=None,
    processed_dir=None,
    patch_size=64,
    neg_ratio=3.0,
    neg_strategy="random",
    candidates_path=None,
    annotations_path=None,
):
    """
    完整预处理流程。

    Parameters
    ----------
    raw_dir       : Path, data/raw/
    processed_dir : Path, data/processed/
    patch_size    : int, 64
    neg_ratio     : float, 负例/正例 最大比例 (random 策略), 0 = 不限制
    neg_strategy  : str, "random" 或 "hard_negative"
    candidates_path : Path
    annotations_path: Path
    """
    raw_dir = raw_dir or RAW_DIR
    processed_dir = processed_dir or PROCESSED_DIR
    patches_dir = processed_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = candidates_path or raw_dir / "candidates.csv"
    annotations_path = annotations_path or raw_dir / "annotations.csv"

    # --- 加载 CSVs ---
    print("=" * 60)
    print("LUNA16 Preprocessing Pipeline")
    print("=" * 60)

    print(f"\n[1/4] Loading data...")
    print(f"  candidates: {candidates_path}")
    print(f"  annotations: {annotations_path}")

    if not candidates_path.exists():
        print(f"  ERROR: candidates.csv not found!")
        return None
    if not annotations_path.exists():
        print(f"  ERROR: annotations.csv not found!")
        return None

    candidates_df = pd.read_csv(candidates_path)
    annotations_df = pd.read_csv(annotations_path)
    candidates_df.columns = candidates_df.columns.str.strip()
    annotations_df.columns = annotations_df.columns.str.strip()

    print(f"  Candidates: {len(candidates_df)} rows, columns: {list(candidates_df.columns)}")
    print(f"  Annotations: {len(annotations_df)} rows, columns: {list(annotations_df.columns)}")

    # --- 建立 subset 映射 ---
    print(f"\n[2/4] Building subset map...")
    subset_map = build_subset_map(raw_dir)

    # --- 遍历所有 subset 目录提取 patches ---
    print(f"\n[3/4] Extracting patches...")
    metadata_rows = []
    subset_dirs = sorted(raw_dir.glob("subset*"))
    subset_dirs = [d for d in subset_dirs if d.is_dir()]

    if not subset_dirs:
        print("  No subset directories found! Expected data/raw/subset0/, subset1/, etc.")
        print("  Please extract the LUNA16 subset zip files first.")
        return None

    print(f"  Found {len(subset_dirs)} subset directories")

    total_pos = 0
    total_neg = 0

    for subset_dir in subset_dirs:
        subset_name = subset_dir.name
        print(f"\n  Processing {subset_name}...")

        # 支持嵌套目录 (subset0/subset0/*.mhd) 和扁平目录 (subset0/*.mhd)
        mhd_files = sorted(subset_dir.glob("*.mhd"))
        if not mhd_files:
            mhd_files = sorted(subset_dir.glob("*/*.mhd"))
        if not mhd_files:
            print(f"    No .mhd files in {subset_name}, skipping")
            continue

        for mhd_path in mhd_files:
            ct, origin, spacing, seriesuid = load_ct_scan(mhd_path)

            ct_cands = candidates_df[candidates_df["seriesuid"] == seriesuid]
            if len(ct_cands) == 0:
                continue

            pos_count = 0
            neg_count = 0

            for idx, (_, row) in enumerate(ct_cands.iterrows()):
                world_xyz = (float(row["coordX"]), float(row["coordY"]), float(row["coordZ"]))

                # 判断正负例
                is_pos = match_candidate_to_annotation(world_xyz, annotations_df, seriesuid)
                class_label = 1 if is_pos else 0

                # 负例采样控制
                if class_label == 0 and neg_ratio > 0:
                    # 先收集，后面统一采样
                    pass

                # 坐标转换
                voxel_xyz = world_to_voxel(world_xyz, origin, spacing)
                x, y, z = voxel_xyz
                max_z, max_y, max_x = ct.shape

                if not (0 <= x < max_x and 0 <= y < max_y and 0 <= z < max_z):
                    continue

                # 提取 patch
                patch = extract_three_layer_patch(ct, voxel_xyz, patch_size)

                # HU 统计 (中心层)
                half = patch_size // 2
                z_start = max(0, z - 1)
                z_end = min(max_z, z + 2)
                y_start = max(0, y - half)
                y_end = min(max_y, y + half)
                x_start = max(0, x - half)
                x_end = min(max_x, x + half)
                hu_region = ct[z_start:z_end, y_start:y_end, x_start:x_end]

                hu_mean = float(np.mean(hu_region))
                hu_std = float(np.std(hu_region))

                # 保存 patch
                subset_id = subset_map.get(seriesuid, -1)
                patch_filename = f"{seriesuid}_{idx:04d}_class{class_label}.npy"
                patch_path = patches_dir / patch_filename
                np.save(patch_path, patch)

                metadata_rows.append({
                    "seriesuid": seriesuid,
                    "subset_id": subset_id,
                    "patch_file": patch_filename,
                    "class": class_label,
                    "world_x": world_xyz[0],
                    "world_y": world_xyz[1],
                    "world_z": world_xyz[2],
                    "voxel_x": voxel_xyz[0],
                    "voxel_y": voxel_xyz[1],
                    "voxel_z": voxel_xyz[2],
                    "hu_mean": round(hu_mean, 2),
                    "hu_std": round(hu_std, 2),
                    "split": "",
                })

                if class_label == 1:
                    pos_count += 1
                else:
                    neg_count += 1

            total_pos += pos_count
            total_neg += neg_count

    print(f"\n  Total patches extracted: {len(metadata_rows)}")
    print(f"  Positives: {total_pos}")
    print(f"  Negatives: {total_neg}")

    # --- 负例采样 ---
    if neg_ratio > 0 and total_neg > total_pos * neg_ratio:
        print(f"\n[4/4] Negative sampling (ratio 1:{neg_ratio})...")
        metadata_df = pd.DataFrame(metadata_rows)
        pos_df = metadata_df[metadata_df["class"] == 1]
        neg_df = metadata_df[metadata_df["class"] == 0]

        target_neg = int(len(pos_df) * neg_ratio)

        if neg_strategy == "random":
            neg_sampled = neg_df.sample(n=target_neg, random_state=42)
        elif neg_strategy == "hard_negative":
            # hard negative: 优先取 HU 均值接近软组织范围 (-200 to 200) 的
            # 因为这更可能是被误检的结节
            neg_df_copy = neg_df.copy()
            soft_tissue_center = -600
            neg_df_copy["hu_score"] = -np.abs(neg_df_copy["hu_mean"] - soft_tissue_center)
            neg_sampled = neg_df_copy.nlargest(target_neg, "hu_score").drop(columns=["hu_score"])
        else:
            neg_sampled = neg_df.sample(n=target_neg, random_state=42)

        metadata_df = pd.concat([pos_df, neg_sampled], ignore_index=True)

        # 删除被丢弃的负例 patch 文件
        kept_files = set(metadata_df["patch_file"])
        for _, row in neg_df.iterrows():
            if row["patch_file"] not in kept_files:
                fp = patches_dir / row["patch_file"]
                if fp.exists():
                    fp.unlink()
        print(f"  After sampling: {len(metadata_df)} total ({len(pos_df)} pos, {len(neg_sampled)} neg)")
    else:
        metadata_df = pd.DataFrame(metadata_rows)

    # --- 保存 metadata ---
    metadata_path = processed_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\n  Metadata saved to {metadata_path}")
    print(f"  Total rows: {len(metadata_df)}")
    print(f"  Columns: {list(metadata_df.columns)}")

    return metadata_df


# ==================== 可视化 ====================

def plot_sample_grid(patches_dir, metadata_df, output_path, n_positive=16, n_negative=16):
    """绘制正负例样本网格"""
    pos_df = metadata_df[metadata_df["class"] == 1]
    neg_df = metadata_df[metadata_df["class"] == 0]

    n_pos_show = min(n_positive, len(pos_df))
    n_neg_show = min(n_negative, len(neg_df))

    pos_samples = pos_df.sample(n=n_pos_show, random_state=42) if n_pos_show > 0 else pos_df.iloc[:0]
    neg_samples = neg_df.sample(n=n_neg_show, random_state=42) if n_neg_show > 0 else neg_df.iloc[:0]

    n_cols = 8
    n_rows = (n_pos_show + n_neg_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.5))
    axes = axes.flatten()

    for i, (_, row) in enumerate(pos_samples.iterrows()):
        patch = np.load(patches_dir / row["patch_file"])
        axes[i].imshow(patch[1], cmap="gray", vmin=0, vmax=1)  # 中心层
        axes[i].set_title(f"POS\n{row['seriesuid'][:8]}", fontsize=7)
        axes[i].axis("off")

    for i, (_, row) in enumerate(neg_samples.iterrows()):
        idx = n_pos_show + i
        patch = np.load(patches_dir / row["patch_file"])
        axes[idx].imshow(patch[1], cmap="gray", vmin=0, vmax=1)
        axes[idx].set_title(f"NEG\n{row['seriesuid'][:8]}", fontsize=7, color="red")
        axes[idx].axis("off")

    for j in range(n_pos_show + n_neg_show, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"LUNA16 Patch Samples (Positive: {n_pos_show}, Negative: {n_neg_show})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample grid saved to {output_path}")


def plot_hu_distribution(metadata_df, output_path):
    """绘制正负例 HU 值分布对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pos_hu = metadata_df[metadata_df["class"] == 1]["hu_mean"]
    neg_hu = metadata_df[metadata_df["class"] == 0]["hu_mean"]

    bins = np.linspace(-1000, 500, 60)

    axes[0].hist(pos_hu, bins=bins, alpha=0.7, color="red", label=f"Positive (n={len(pos_hu)})")
    axes[0].hist(neg_hu, bins=bins, alpha=0.7, color="blue", label=f"Negative (n={len(neg_hu)})")
    axes[0].set_xlabel("Mean HU")
    axes[0].set_ylabel("Count")
    axes[0].set_title("HU Distribution: Positive vs Negative Candidates")
    axes[0].legend()
    axes[0].axvline(x=-600, color="gray", linestyle="--", alpha=0.5, label="Lung window center")
    axes[0].axvline(x=-200, color="green", linestyle="--", alpha=0.5, label="Soft tissue")
    axes[0].legend()

    axes[1].hist(pos_hu, bins=bins, alpha=0.7, color="red", density=True, label="Positive")
    axes[1].hist(neg_hu, bins=bins, alpha=0.7, color="blue", density=True, label="Negative")
    axes[1].set_xlabel("Mean HU")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Normalized HU Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"HU distribution plot saved to {output_path}")


def plot_class_imbalance(metadata_df, output_path):
    """类别不均衡可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. 正负样本总数
    counts = metadata_df["class"].value_counts().sort_index()
    axes[0].bar(["Negative (0)", "Positive (1)"], [counts.get(0, 0), counts.get(1, 0)],
                color=["blue", "red"], edgecolor="black")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Class Distribution")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        axes[0].text(i, v + max(counts) * 0.02, str(v), ha="center", fontweight="bold")

    # 2. 每例候选数分布
    per_uid = metadata_df.groupby("seriesuid").size()
    axes[1].hist(per_uid, bins=30, color="green", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Candidates per CT")
    axes[1].set_ylabel("Number of CTs")
    axes[1].set_title(f"Candidates per Scan (median={per_uid.median():.0f})")

    # 3. 正负例占比饼图
    axes[2].pie([counts.get(0, 0), counts.get(1, 0)],
                labels=["Negative", "Positive"],
                colors=["blue", "red"],
                autopct="%1.1f%%",
                explode=(0, 0.05),
                startangle=90)
    axes[2].set_title("Class Ratio")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Class imbalance plot saved to {output_path}")


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="LUNA16 Preprocessing Pipeline")
    parser.add_argument("--raw_dir", type=str, default="data/raw",
                        help="Path to data/raw/")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="Path to data/processed/")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--neg_ratio", type=float, default=3.0,
                        help="Max negative/positive ratio (0=no limit)")
    parser.add_argument("--neg_strategy", type=str, default="random",
                        choices=["random", "hard_negative"])
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--print_meta", type=int, default=0,
                        help="Print first N rows of metadata")

    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / args.raw_dir
    processed_dir = PROJECT_ROOT / args.processed_dir
    patches_dir = processed_dir / "patches"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metadata_df = run_preprocessing(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        patch_size=args.patch_size,
        neg_ratio=args.neg_ratio,
        neg_strategy=args.neg_strategy,
    )

    if metadata_df is None or len(metadata_df) == 0:
        print("\nNo data processed. Make sure LUNA16 data is in data/raw/")
        return

    if args.print_meta > 0:
        print(f"\n=== First {min(args.print_meta, len(metadata_df))} rows of metadata ===")
        print(metadata_df.head(args.print_meta).to_string())

    if args.visualize:
        print("\nGenerating visualizations...")
        plot_sample_grid(patches_dir, metadata_df,
                         str(FIGURES_DIR / "patch_samples_grid.png"))
        plot_hu_distribution(metadata_df,
                             str(FIGURES_DIR / "hu_distribution.png"))
        plot_class_imbalance(metadata_df,
                             str(FIGURES_DIR / "class_imbalance.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
