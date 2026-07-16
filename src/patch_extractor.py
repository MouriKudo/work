"""
LUNA16 CT Patch 提取器
======================
从 .mhd/.raw 文件中读取 CT 扫描，根据 candidates.csv 的坐标
裁剪 64x64 的三层 patch (z-1, z, z+1)。

每个 patch 输出 shape: (3, 64, 64)
"""

import os
import re
import sys
import io
import argparse
from pathlib import Path
from glob import glob
from collections import defaultdict

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 需要先安装 SimpleITK: pip install SimpleITK
try:
    import SimpleITK as sitk
except ImportError:
    print("ERROR: SimpleITK not installed. Run: pip install SimpleITK")
    sys.exit(1)

from coordinate_utils import (
    world_to_voxel,
    extract_patch_bounds,
    match_candidate_to_annotation,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def lung_window(image_hu, width=1500, level=-600):
    """
    肺窗窗宽窗位归一化。

    公式:
        hu_min = level - width / 2
        hu_max = level + width / 2
        裁剪到 [hu_min, hu_max]，再归一化到 [0, 1]

    Parameters
    ----------
    image_hu : np.ndarray, 原始 HU 值
    width    : float, 窗宽 (默认 1500)
    level    : float, 窗位 (默认 -600)

    Returns
    -------
    normalized : np.ndarray, float32, 范围 [0, 1]
    """
    hu_min = level - width / 2.0  # -1350
    hu_max = level + width / 2.0  # 150
    image = np.clip(image_hu, hu_min, hu_max)
    image = (image - hu_min) / (hu_max - hu_min)
    return image.astype(np.float32)


def load_ct_scan(mhd_path):
    """
    用 SimpleITK 读取 CT 扫描。

    Returns
    -------
    ct_array    : np.ndarray, shape (z, y, x), uint16 或 float
    origin      : (x, y, z)
    spacing     : (x, y, z)
    seriesuid   : str
    """
    image = sitk.ReadImage(mhd_path)
    ct_array = sitk.GetArrayFromImage(image)  # shape: (z, y, x)
    origin = image.GetOrigin()  # (x, y, z)
    spacing = image.GetSpacing()  # (x, y, z)
    # 从文件名提取 seriesuid
    seriesuid = os.path.splitext(os.path.basename(mhd_path))[0]
    return ct_array, origin, spacing, seriesuid


def extract_three_layer_patch(ct_array, voxel_xyz, patch_size=64):
    """
    裁剪三层 64x64 patch。
    三层 = (z-1, z, z+1)，如果边界不够则重复一层。

    Parameters
    ----------
    ct_array   : np.ndarray, shape (z, y, x)
    voxel_xyz  : (x, y, z) — 中心体素索引
    patch_size : int, patch 边长

    Returns
    -------
    patch : np.ndarray, shape (3, patch_size, patch_size), float32, [0,1] 归一化
    """
    max_z, max_y, max_x = ct_array.shape
    half = patch_size // 2
    x, y, z = voxel_xyz

    # x, y 边界
    x_start = max(0, x - half)
    y_start = max(0, y - half)
    x_end = min(max_x, x_start + patch_size)
    y_end = min(max_y, y_start + patch_size)
    # 如果需要，回退 x_start / y_start
    x_start = x_end - patch_size
    y_start = y_end - patch_size
    if x_start < 0:
        x_start = 0
        x_end = patch_size
    if y_start < 0:
        y_start = 0
        y_end = patch_size

    layers = []
    for dz in [-1, 0, 1]:
        z_idx = z + dz
        z_idx = max(0, min(max_z - 1, z_idx))
        layer = ct_array[z_idx, y_start:y_end, x_start:x_end].astype(np.float32)
        # 肺窗归一化
        layer = lung_window(layer)
        layers.append(layer)

    patch = np.stack(layers, axis=0)  # (3, 64, 64)
    return patch


def load_candidates(csv_path):
    """读取 candidates.csv, 返回 DataFrame"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"  Loaded {len(df)} candidates from {csv_path}")
    print(f"  Columns: {list(df.columns)}")
    return df


def load_annotations(csv_path):
    """读取 annotations.csv, 返回 DataFrame"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"  Loaded {len(df)} annotations from {csv_path}")
    print(f"  Columns: {list(df.columns)}")
    return df


def determine_subset(seriesuid, subset_map):
    """
    根据 seriesuid 确定属于哪个 subset。
    subset_map: {seriesuid: subset_id}
    """
    return subset_map.get(seriesuid, -1)


def build_subset_map(subset_dirs):
    """
    遍历所有 subset 目录，建立 {seriesuid: subset_id} 映射。

    Parameters
    ----------
    subset_dirs : list of Path, 如 [data/raw/subset0/, data/raw/subset1/, ...]

    Returns
    -------
    subset_map : dict
    """
    subset_map = {}
    for sd in subset_dirs:
        if not sd.exists():
            continue
        subset_name = sd.name
        try:
            subset_id = int(re.search(r"(\d+)", subset_name).group(1))
        except (AttributeError, ValueError):
            continue
        mhd_files = glob(str(sd / "*.mhd"))
        for mf in mhd_files:
            seriesuid = os.path.splitext(os.path.basename(mf))[0]
            subset_map[seriesuid] = subset_id
    return subset_map


def extract_patches_from_subset(
    ct_dir,
    candidates_df,
    annotations_df,
    output_dir,
    patch_size=64,
    subset_map=None,
):
    """
    对一个 subset 目录中的所有 CT 提取 patches。

    Parameters
    ----------
    ct_dir          : Path to subset directory containing .mhd/.raw files
    candidates_df   : DataFrame from candidates.csv
    annotations_df  : DataFrame from annotations.csv
    output_dir      : Path, where .npy patches will be saved
    patch_size      : int
    subset_map      : dict, {seriesuid: subset_id}

    Returns
    -------
    metadata_rows : list of dicts for metadata.csv
    """
    mhd_files = sorted(glob(str(ct_dir / "*.mhd")))
    if not mhd_files:
        print(f"  No .mhd files found in {ct_dir}")
        return []

    metadata_rows = []
    os.makedirs(output_dir, exist_ok=True)

    for mhd_path in mhd_files:
        try:
            ct_array, origin, spacing, seriesuid = load_ct_scan(mhd_path)
            print(f"  Processing {seriesuid}: shape={ct_array.shape}")
        except Exception as e:
            print(f"  ERROR loading {mhd_path}: {e}")
            continue

        # 找到属于这个 CT 的所有 candidates
        ct_candidates = candidates_df[candidates_df["seriesuid"] == seriesuid]
        if len(ct_candidates) == 0:
            print(f"    No candidates for {seriesuid}, skipping")
            continue

        for idx, (_, row) in enumerate(ct_candidates.iterrows()):
            world_xyz = (float(row["coordX"]), float(row["coordY"]), float(row["coordZ"]))
            class_label = row.get("class", None)
            # 如果没有 class 列，从 annotations 匹配
            if class_label is None or pd.isna(class_label):
                is_pos = match_candidate_to_annotation(
                    world_xyz, annotations_df, seriesuid
                )
                class_label = 1 if is_pos else 0

            voxel_xyz = world_to_voxel(world_xyz, origin, spacing)

            # 边界检查
            max_z, max_y, max_x = ct_array.shape
            x, y, z = voxel_xyz
            if not (0 <= x < max_x and 0 <= y < max_y and 0 <= z < max_z):
                print(f"    Candidate {idx}: out of bounds, skipping")
                continue

            # 提取三层 patch
            patch = extract_three_layer_patch(ct_array, voxel_xyz, patch_size)

            # 计算 patch 内的 HU 统计
            hu_patch = ct_array[
                max(0, z - 1): min(max_z, z + 2),
                max(0, y - patch_size // 2): min(max_y, y + patch_size // 2),
                max(0, x - patch_size // 2): min(max_x, x + patch_size // 2),
            ]
            hu_mean = float(np.mean(hu_patch))
            hu_std = float(np.std(hu_patch))

            # 保存 patch
            subset_id = determine_subset(seriesuid, subset_map or {})
            patch_filename = f"{seriesuid}_{idx:04d}_class{class_label}.npy"
            patch_path = output_dir / patch_filename
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
                "hu_mean": hu_mean,
                "hu_std": hu_std,
                "split": "",
            })

    print(f"    Extracted {len(metadata_rows)} patches from {ct_dir.name}")
    return metadata_rows


def visualize_sample_patches(patches_dir, metadata_df, output_path, n_positive=8, n_negative=8):
    """
    可视化正例和负例 patch 的网格图。

    Parameters
    ----------
    patches_dir : Path, patches 存储目录
    metadata_df : DataFrame, 从 metadata.csv 读取
    output_path : str, 输出图片路径
    n_positive   : 正例展示数量
    n_negative   : 负例展示数量
    """
    pos_df = metadata_df[metadata_df["class"] == 1].sample(
        min(n_positive, (metadata_df["class"] == 1).sum()), random_state=42
    )
    neg_df = metadata_df[metadata_df["class"] == 0].sample(
        min(n_negative, (metadata_df["class"] == 0).sum()), random_state=42
    )

    total = n_positive + n_negative
    fig, axes = plt.subplots(total, 3, figsize=(9, 2 * total))

    for i, (_, row) in enumerate(pos_df.iterrows()):
        patch = np.load(patches_dir / row["patch_file"])
        for j in range(3):
            axes[i][j].imshow(patch[j], cmap="gray", vmin=0, vmax=1)
            axes[i][j].axis("off")
        axes[i][0].set_ylabel(f"Pos\n{row['seriesuid'][:12]}", fontsize=7, rotation=0, labelpad=40)

    for i, (_, row) in enumerate(neg_df.iterrows()):
        patch = np.load(patches_dir / row["patch_file"])
        for j in range(3):
            axes[i + n_positive][j].imshow(patch[j], cmap="gray", vmin=0, vmax=1)
            axes[i + n_positive][j].axis("off")
        axes[i + n_positive][0].set_ylabel(
            f"Neg\n{row['seriesuid'][:12]}", fontsize=7, rotation=0, labelpad=40
        )

    axes[0][0].set_title("Slice z-1", fontsize=9)
    axes[0][1].set_title("Slice z (center)", fontsize=9)
    axes[0][2].set_title("Slice z+1", fontsize=9)

    fig.suptitle("LUNA16 Patch Samples (64x64, 3 slices)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Patch visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LUNA16 Patch Extractor")
    parser.add_argument("--subset_dir", type=str, default="data/raw/subset0",
                        help="Path to subset directory containing .mhd files")
    parser.add_argument("--candidates", type=str, default="data/raw/candidates.csv",
                        help="Path to candidates.csv")
    parser.add_argument("--annotations", type=str, default="data/raw/annotations.csv",
                        help="Path to annotations.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed/patches",
                        help="Directory to save .npy patch files")
    parser.add_argument("--metadata", type=str, default="data/processed/metadata.csv",
                        help="Path to output metadata.csv")
    parser.add_argument("--patch_size", type=int, default=64,
                        help="Patch size (default 64)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate sample visualization grid")
    parser.add_argument("--viz_output", type=str, default="paper_figs/patch_samples.png",
                        help="Visualization output path")

    args = parser.parse_args()

    import pandas as pd

    # 转为 Project 根目录下的绝对路径
    subset_dir = PROJECT_ROOT / args.subset_dir
    candidates_path = PROJECT_ROOT / args.candidates
    annotations_path = PROJECT_ROOT / args.annotations
    output_dir = PROJECT_ROOT / args.output_dir
    metadata_path = PROJECT_ROOT / args.metadata
    viz_path = PROJECT_ROOT / args.viz_output

    # 检查文件
    if not subset_dir.exists():
        print(f"ERROR: Subset directory not found: {subset_dir}")
        print("Download LUNA16 data first, then extract subset0.zip to data/raw/subset0/")
        return

    if not candidates_path.exists():
        print(f"ERROR: candidates.csv not found: {candidates_path}")
        return

    if not annotations_path.exists():
        print(f"ERROR: annotations.csv not found: {annotations_path}")
        return

    print("=" * 60)
    print("LUNA16 Patch Extractor")
    print("=" * 60)
    print(f"  Subset dir: {subset_dir}")
    print(f"  Candidates: {candidates_path}")
    print(f"  Annotations: {annotations_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Patch size: {args.patch_size}")
    print("=" * 60)

    # 加载数据
    candidates_df = load_candidates(candidates_path)
    annotations_df = load_annotations(annotations_path)

    # 提取 patches
    print("\nExtracting patches...")
    rows = extract_patches_from_subset(
        subset_dir, candidates_df, annotations_df,
        output_dir, args.patch_size
    )

    if rows:
        metadata_df = pd.DataFrame(rows)
        metadata_df.to_csv(metadata_path, index=False)
        print(f"\nMetadata saved to {metadata_path} ({len(metadata_df)} rows)")

        # 统计
        n_pos = (metadata_df["class"] == 1).sum()
        n_neg = (metadata_df["class"] == 0).sum()
        print(f"  Positive samples: {n_pos}")
        print(f"  Negative samples: {n_neg}")
        print(f"  Pos/Neg ratio: {n_pos / max(1, n_neg):.2f}")

        # 可视化
        if args.visualize:
            os.makedirs(viz_path.parent, exist_ok=True)
            visualize_sample_patches(output_dir, metadata_df, str(viz_path))
    else:
        print("No patches extracted.")


if __name__ == "__main__":
    main()
