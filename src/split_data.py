"""
LUNA16 Train/Val/Test 划分
===========================
按 seriesuid (患者粒度) 划分数据集。
策略: subsets 0-7 → train, subset 8 → val, subset 9 → test

产出:
  - 数据统计表
  - 类别不均衡图
  - 样本网格图
  - 更新 metadata.csv (写入 split 字段)
"""

import os
import sys
import io
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PATCHES_DIR = PROCESSED_DIR / "patches"
FIGURES_DIR = PROJECT_ROOT / "paper_figs"


def assign_split(metadata_df, train_subsets, val_subsets, test_subsets):
    """
    根据 subset_id 分配 split 标签。
    同一患者的所有候选必定属于同一个 split（因为 seriesuid 只在一个 subset 里）。
    """
    df = metadata_df.copy()

    train_mask = df["subset_id"].isin(train_subsets)
    val_mask = df["subset_id"].isin(val_subsets)
    test_mask = df["subset_id"].isin(test_subsets)

    df.loc[train_mask, "split"] = "train"
    df.loc[val_mask, "split"] = "val"
    df.loc[test_mask, "split"] = "test"

    unassigned = df[df["split"] == ""]
    if len(unassigned) > 0:
        print(f"  WARNING: {len(unassigned)} samples not assigned to any split")
        print(f"  Unassigned subset_ids: {unassigned['subset_id'].unique()}")

    return df


def print_statistics(df):
    """打印详细统计"""
    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)

    # 总体统计
    total = len(df)
    n_pos = (df["class"] == 1).sum()
    n_neg = (df["class"] == 0).sum()
    n_cts = df["seriesuid"].nunique()

    print(f"\n  Total samples:     {total}")
    print(f"  Unique CT scans:   {n_cts}")
    print(f"  Positive samples:  {n_pos} ({n_pos / max(1, total) * 100:.1f}%)")
    print(f"  Negative samples:  {n_neg} ({n_neg / max(1, total) * 100:.1f}%)")
    print(f"  Pos/Neg ratio:     {n_pos / max(1, n_neg):.2f}")

    # 每例候选分布
    cands_per_ct = df.groupby("seriesuid").size()
    print(f"\n  Candidates per CT:")
    print(f"    Min:    {cands_per_ct.min()}")
    print(f"    Median: {cands_per_ct.median():.0f}")
    print(f"    Mean:   {cands_per_ct.mean():.1f}")
    print(f"    Max:    {cands_per_ct.max()}")

    # HU 统计
    print(f"\n  HU distribution:")
    for label, name in zip([0, 1], ["Negative", "Positive"]):
        sub = df[df["class"] == label]["hu_mean"]
        if len(sub) > 0:
            print(f"    {name}: mean={sub.mean():.1f}, std={sub.std():.1f}, "
                  f"min={sub.min():.0f}, max={sub.max():.0f}")

    # 按 split 统计
    print(f"\n  Split breakdown:")
    print(f"  {'Split':<8} {'Total':<8} {'Pos':<8} {'Neg':<8} {'CTs':<8} {'Ratio':<8}")
    print(f"  {'-' * 48}")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        pos = (sub["class"] == 1).sum()
        neg = (sub["class"] == 0).sum()
        cts = sub["seriesuid"].nunique()
        ratio = pos / max(1, neg)
        print(f"  {split:<8} {len(sub):<8} {pos:<8} {neg:<8} {cts:<8} {ratio:<8.2f}")

    print("=" * 60)


def plot_split_distribution(df, output_path):
    """可视化各 split 的分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    splits = ["train", "val", "test"]
    colors = {"train": "green", "val": "orange", "test": "red"}

    # 1. 样本量
    counts = df["split"].value_counts()
    axes[0, 0].bar([s for s in splits if s in counts.index],
                   [counts.get(s, 0) for s in splits if s in counts.index],
                   color=[colors[s] for s in splits if s in counts.index],
                   edgecolor="black")
    axes[0, 0].set_title("Samples per Split")
    axes[0, 0].set_ylabel("Count")

    # 2. 正负例分布
    x = range(len(splits))
    width = 0.35
    pos_counts = [(df[(df["split"] == s) & (df["class"] == 1)]) for s in splits]
    neg_counts = [(df[(df["split"] == s) & (df["class"] == 0)]) for s in splits]

    active_splits = [s for s in splits if s in counts.index]
    x_pos = range(len(active_splits))

    axes[0, 1].bar(x_pos, [len(p) for p in pos_counts if len(p) >= 0],
                   width, label="Positive", color="red", edgecolor="black")
    axes[0, 1].bar([p + width for p in x_pos],
                   [len(n) for n in neg_counts],
                   width, label="Negative", color="blue", edgecolor="black")
    axes[0, 1].set_xticks([p + width / 2 for p in x_pos])
    axes[0, 1].set_xticklabels(active_splits)
    axes[0, 1].set_title("Pos/Neg per Split")
    axes[0, 1].legend()

    # 3. CT 数量
    ct_counts = [df[df["split"] == s]["seriesuid"].nunique() for s in active_splits]
    axes[1, 0].bar(active_splits, ct_counts, color=[colors[s] for s in active_splits],
                   edgecolor="black")
    axes[1, 0].set_title("Unique CT Scans per Split")
    axes[1, 0].set_ylabel("Count")

    # 4. HU 分布对比
    for split in active_splits:
        sub = df[df["split"] == split]
        if len(sub) > 0:
            axes[1, 1].hist(sub["hu_mean"], bins=50, alpha=0.5,
                            color=colors[split], label=split, density=True)
    axes[1, 1].set_xlabel("Mean HU")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("HU Distribution by Split")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Split distribution plot saved to {output_path}")


def plot_candidate_histogram(df, output_path):
    """每例候选数分布直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cands_per_ct = df.groupby("seriesuid").size()

    axes[0].hist(cands_per_ct, bins=40, color="purple", edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Number of Candidates")
    axes[0].set_ylabel("Number of CT Scans")
    axes[0].set_title(f"Candidates per CT Scan (n={len(cands_per_ct)} CTs)")
    axes[0].axvline(cands_per_ct.median(), color="red", linestyle="--",
                    label=f"Median: {cands_per_ct.median():.0f}")
    axes[0].legend()

    axes[1].boxplot(cands_per_ct, vert=True)
    axes[1].set_ylabel("Number of Candidates")
    axes[1].set_title("Boxplot: Candidates per CT")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Candidate histogram saved to {output_path}")


def generate_statistics_table(df, output_path):
    """生成数据统计表 CSV"""
    stats_rows = []

    # 总体统计
    stats_rows.append({"Category": "Overall", "Metric": "Total Samples", "Value": len(df)})
    stats_rows.append({"Category": "Overall", "Metric": "Unique CTs", "Value": df["seriesuid"].nunique()})
    stats_rows.append({"Category": "Overall", "Metric": "Positive", "Value": (df["class"] == 1).sum()})
    stats_rows.append({"Category": "Overall", "Metric": "Negative", "Value": (df["class"] == 0).sum()})
    stats_rows.append({"Category": "Overall", "Metric": "Pos/Neg Ratio",
                       "Value": round((df["class"] == 1).sum() / max(1, (df["class"] == 0).sum()), 3)})

    # HU 统计
    for label, name in [(0, "Negative"), (1, "Positive")]:
        sub = df[df["class"] == label]["hu_mean"]
        if len(sub) > 0:
            stats_rows.append({"Category": f"HU {name}", "Metric": "mean", "Value": round(sub.mean(), 1)})
            stats_rows.append({"Category": f"HU {name}", "Metric": "std", "Value": round(sub.std(), 1)})
            stats_rows.append({"Category": f"HU {name}", "Metric": "min", "Value": round(sub.min(), 1)})
            stats_rows.append({"Category": f"HU {name}", "Metric": "max", "Value": round(sub.max(), 1)})

    # Split 统计
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        stats_rows.append({"Category": f"Split {split}", "Metric": "Total", "Value": len(sub)})
        stats_rows.append({"Category": f"Split {split}", "Metric": "Positive", "Value": (sub["class"] == 1).sum()})
        stats_rows.append({"Category": f"Split {split}", "Metric": "Negative", "Value": (sub["class"] == 0).sum()})
        stats_rows.append({"Category": f"Split {split}", "Metric": "Unique CTs",
                           "Value": sub["seriesuid"].nunique()})

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_path, index=False)
    print(f"Statistics table saved to {output_path}")

    # 同时打印到控制台
    print("\n" + stats_df.to_string(index=False))
    return stats_df


def main():
    parser = argparse.ArgumentParser(description="LUNA16 Train/Val/Test Split")
    parser.add_argument("--metadata", type=str, default="data/processed/metadata.csv",
                        help="Path to metadata.csv")
    parser.add_argument("--train_subsets", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated subset IDs for train")
    parser.add_argument("--val_subsets", type=str, default="8",
                        help="Comma-separated subset IDs for val")
    parser.add_argument("--test_subsets", type=str, default="9",
                        help="Comma-separated subset IDs for test")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--print_meta", type=int, default=100)

    args = parser.parse_args()

    metadata_path = PROJECT_ROOT / args.metadata
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LUNA16 Train/Val/Test Split")
    print("=" * 60)

    if not metadata_path.exists():
        print(f"ERROR: metadata.csv not found at {metadata_path}")
        print("Run preprocessing first: python src/preprocess.py")
        return

    # 解析 subset 分配
    train_subsets = [int(x.strip()) for x in args.train_subsets.split(",")]
    val_subsets = [int(x.strip()) for x in args.val_subsets.split(",")]
    test_subsets = [int(x.strip()) for x in args.test_subsets.split(",")]

    print(f"  Train subsets: {train_subsets}")
    print(f"  Val subsets:   {val_subsets}")
    print(f"  Test subsets:  {test_subsets}")

    # 检查重叠
    all_ids = set(train_subsets + val_subsets + test_subsets)
    if len(all_ids) != len(train_subsets) + len(val_subsets) + len(test_subsets):
        print("  WARNING: Subset overlap detected! Check your assignments.")

    # 加载 metadata
    df = pd.read_csv(metadata_path)
    print(f"\n  Loaded {len(df)} samples from metadata.csv")

    # 分配 split
    print(f"\n[1/4] Assigning splits...")
    df = assign_split(df, train_subsets, val_subsets, test_subsets)

    # 保存
    print(f"\n[2/4] Saving updated metadata...")
    df.to_csv(metadata_path, index=False)
    print(f"  Saved to {metadata_path}")

    # 统计
    print(f"\n[3/4] Generating statistics...")
    print_statistics(df)
    stats_df = generate_statistics_table(df, str(FIGURES_DIR / "statistics.csv"))

    # 可视化
    if args.visualize:
        print(f"\n[4/4] Generating plots...")
        plot_split_distribution(df, str(FIGURES_DIR / "split_distribution.png"))
        plot_candidate_histogram(df, str(FIGURES_DIR / "candidate_histogram.png"))

    # 打印前 N 行
    if args.print_meta > 0:
        n = min(args.print_meta, len(df))
        print(f"\n=== First {n} rows of metadata ===")
        print(df.head(n).to_string())

    print("\nDone!")
    print(f"  Metadata: {metadata_path}")
    print(f"  Figures:  {FIGURES_DIR}")
    print(f"  Patches:  {PATCHES_DIR}")


if __name__ == "__main__":
    main()
