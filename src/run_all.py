"""All visualizations using cv2 + PIL instead of matplotlib savefig"""
import os, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

PROJECT_ROOT = Path('D:/luna16-work')
PATCHES_DIR = PROJECT_ROOT / 'data' / 'processed' / 'patches'
METADATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'metadata.csv'
FIGURES_DIR = PROJECT_ROOT / 'paper_figs'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(METADATA_PATH))
print(f'Loaded {len(df)} rows | Pos: {(df["class"]==1).sum()}, Neg: {(df["class"]==0).sum()}')

# ============================================================
# 1. Patch Sample Grid (cv2)
# ============================================================
print("\n[1/6] patch_samples_grid.png...")
try:
    pos_df = df[df['class'] == 1].sample(n=min(16, (df['class']==1).sum()), random_state=42)
    neg_df = df[df['class'] == 0].sample(n=min(16, (df['class']==0).sum()), random_state=42)

    patches_list = []
    labels = []
    for _, row in pos_df.iterrows():
        p = np.load(PATCHES_DIR / row['patch_file'])
        patches_list.append(p[1])  # center slice
        labels.append('POS')
    for _, row in neg_df.iterrows():
        p = np.load(PATCHES_DIR / row['patch_file'])
        patches_list.append(p[1])
        labels.append('NEG')

    # Build grid: 8 cols, N rows
    n = len(patches_list)
    n_cols = 8
    n_rows = (n + n_cols - 1) // n_cols
    cell_size = 64
    margin = 4
    label_h = 16

    grid_h = n_rows * (cell_size + margin + label_h) + margin
    grid_w = n_cols * (cell_size + margin) + margin
    grid = np.ones((grid_h, grid_w), dtype=np.float32)

    for i, (patch_img, label) in enumerate(zip(patches_list, labels)):
        r = i // n_cols
        c = i % n_cols
        y_start = margin + r * (cell_size + margin + label_h)
        x_start = margin + c * (cell_size + margin)
        grid[y_start:y_start+cell_size, x_start:x_start+cell_size] = patch_img

        # Add label text as a colored bar at top
        if label == 'POS':
            grid[y_start:y_start+3, x_start:x_start+cell_size] = 1.0
        else:
            grid[y_start:y_start+3, x_start:x_start+cell_size] = 0.2

    # Convert to uint8 and save
    grid_uint8 = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(FIGURES_DIR / 'patch_samples_grid.png'), grid_uint8)
    print("  patch_samples_grid.png OK")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================
# 2. HU Distribution — use ASCII art + statistics print instead
# (We'll skip matplotlib-dependent plots and create summary PNGs with cv2)
# ============================================================
print("\n[2/6] hu_distribution.png...")
try:
    pos_hu = df[df['class'] == 1]['hu_mean'].values
    neg_hu = df[df['class'] == 0]['hu_mean'].values

    # Create histogram manually and render with cv2
    h = 400; w = 800; margin = 60
    img = np.ones((h, w, 3), dtype=np.float32)

    bins = np.linspace(-1000, 500, 61)
    pos_hist, _ = np.histogram(pos_hu, bins=bins)
    neg_hist, _ = np.histogram(neg_hu, bins=bins)

    max_count = max(pos_hist.max(), neg_hist.max())
    plot_h = h - 2 * margin
    plot_w = w - 2 * margin
    bar_w = plot_w // len(pos_hist)

    for i in range(len(pos_hist)):
        x0 = margin + i * bar_w
        # Positive (red)
        h_pos = int(pos_hist[i] / max_count * plot_h)
        if h_pos > 0:
            cv2.rectangle(img, (x0, h - margin - h_pos), (x0 + bar_w//2, h - margin), (1, 0, 0), -1)
        # Negative (blue)
        h_neg = int(neg_hist[i] / max_count * plot_h)
        if h_neg > 0:
            cv2.rectangle(img, (x0 + bar_w//2, h - margin - h_neg), (x0 + bar_w, h - margin), (0, 0, 1), -1)

    # Legend
    cv2.rectangle(img, (w-160, 10), (w-145, 20), (1, 0, 0), -1)
    cv2.rectangle(img, (w-160, 30), (w-145, 40), (0, 0, 1), -1)

    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]  # RGB -> BGR
    cv2.imwrite(str(FIGURES_DIR / 'hu_distribution.png'), img_uint8)
    print("  hu_distribution.png OK")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================
# 3. Class Imbalance (cv2 bar chart)
# ============================================================
print("\n[3/6] class_imbalance.png...")
try:
    counts = df['class'].value_counts().sort_index()
    n0 = counts.get(0, 0)
    n1 = counts.get(1, 0)

    h = 400; w = 600; margin = 80
    img = np.ones((h, w, 3), dtype=np.float32)

    bar_w = 100; bar_gap = 80
    max_val = max(n0, n1) * 1.15
    plot_h = h - 2 * margin

    for i, (val, color, label) in enumerate([(n0, (0, 0, 1), 'Neg'), (n1, (1, 0, 0), 'Pos')]):
        x0 = margin + i * (bar_w + bar_gap) + 50
        bar_h = int(val / max_val * plot_h)
        cv2.rectangle(img, (x0, h - margin - bar_h), (x0 + bar_w, h - margin), color, -1)
        cv2.rectangle(img, (x0, h - margin - bar_h), (x0 + bar_w, h - margin), (0, 0, 0), 1)

    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(str(FIGURES_DIR / 'class_imbalance.png'), img_uint8)
    print("  class_imbalance.png OK")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================
# 4. Split Distribution (cv2 bar chart)
# ============================================================
print("\n[4/6] split_distribution.png...")
try:
    h = 400; w = 900; margin = 80
    img = np.ones((h, w, 3), dtype=np.float32)

    splits = ['train', 'val', 'test']
    colors = [(0, 0.6, 0), (1, 0.6, 0), (1, 0, 0)]  # green, orange, red
    values = [(df['split'] == s).sum() for s in splits]

    max_val = max(max(values), 1) * 1.2
    bar_w = 80; bar_gap = 60; group_gap = 100
    plot_h = h - 2 * margin

    for i, (val, color, label) in enumerate(zip(values, colors, splits)):
        x0 = margin + i * (bar_w + bar_gap + group_gap) + 50
        bar_h = int(val / max_val * plot_h)
        cv2.rectangle(img, (x0, h - margin - bar_h), (x0 + bar_w, h - margin), color, -1)
        cv2.rectangle(img, (x0, h - margin - bar_h), (x0 + bar_w, h - margin), (0, 0, 0), 1)

    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(str(FIGURES_DIR / 'split_distribution.png'), img_uint8)
    print("  split_distribution.png OK")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================
# 5. Candidate Histogram
# ============================================================
print("\n[5/6] candidate_histogram.png...")
try:
    cands_per_ct = df.groupby('seriesuid').size().values

    h = 400; w = 700; margin = 60
    img = np.ones((h, w, 3), dtype=np.float32)

    hist, bin_edges = np.histogram(cands_per_ct, bins=40)
    max_h = hist.max()
    plot_h = h - 2 * margin; plot_w = w - 2 * margin
    bar_w = plot_w // len(hist)

    for i, val in enumerate(hist):
        x0 = margin + i * bar_w
        bar_h = int(val / max_h * plot_h)
        cv2.rectangle(img, (x0, h - margin - bar_h), (x0 + bar_w - 1, h - margin), (0.5, 0, 0.5), -1)

    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(str(FIGURES_DIR / 'candidate_histogram.png'), img_uint8)
    print("  candidate_histogram.png OK")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================
# 6. Statistics Table + Split Assignment
# ============================================================
print("\n[6/6] statistics.csv + metadata update...")

# Fix split column type
if 'split' in df.columns:
    df['split'] = df['split'].astype(str)
    df.loc[df['split'] == 'nan', 'split'] = ''
    df.loc[df['split'] == '', 'split'] = 'train'  # subset0 -> train

# Save metadata — save to timestamped file if locked
metadata_saved = False
for attempt in range(3):
    save_path = METADATA_PATH if attempt == 2 else METADATA_PATH.parent / f'metadata_{attempt}.csv'
    try:
        df.to_csv(str(save_path), index=False)
        if save_path != METADATA_PATH:
            print(f"  Note: saved to {save_path.name} (metadata.csv locked)")
            print(f"  Close any programs holding metadata.csv, then run:")
            print(f"    move /Y data\\processed\\metadata_0.csv data\\processed\\metadata.csv")
        metadata_saved = True
        break
    except (PermissionError, OSError) as e:
        if attempt == 2:
            print(f"  WARNING: Could not save metadata: {e}")
            print(f"  Please close Excel/VSCode and re-run.")
        continue
if not metadata_saved:
    # Last resort: save to a completely new name
    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    alt_path = METADATA_PATH.parent / f'metadata_{ts}.csv'
    df.to_csv(str(alt_path), index=False)
    print(f"  Saved to {alt_path.name}")
    print(f"  After closing metadata.csv, rename this file to metadata.csv")
print(f"  metadata rows: {len(df)}")

# Statistics
total = len(df)
n_pos = (df['class'] == 1).sum()
n_neg = (df['class'] == 0).sum()
n_cts = df['seriesuid'].nunique()

# Print statistics
from math import floor, ceil
print("\n" + "=" * 60)
print("DATA STATISTICS")
print("=" * 60)
print(f"  Total samples:     {total}")
print(f"  Unique CT scans:   {n_cts}")
print(f"  Positive samples:  {n_pos} ({n_pos/max(1,total)*100:.1f}%)")
print(f"  Negative samples:  {n_neg} ({n_neg/max(1,total)*100:.1f}%)")
print(f"  Pos/Neg ratio:     {n_pos/max(1,n_neg):.2f}")

cands_per_ct_series = df.groupby('seriesuid').size()
print(f"\n  Candidates per CT:")
print(f"    Min:    {cands_per_ct_series.min()}")
print(f"    Median: {cands_per_ct_series.median():.0f}")
print(f"    Mean:   {cands_per_ct_series.mean():.1f}")
print(f"    Max:    {cands_per_ct_series.max()}")

print(f"\n  HU distribution:")
for label, name in zip([0, 1], ['Negative', 'Positive']):
    sub = df[df['class'] == label]['hu_mean']
    if len(sub) > 0:
        print(f"    {name}: mean={sub.mean():.1f}, std={sub.std():.1f}, min={sub.min():.0f}, max={sub.max():.0f}")

print(f"\n  Split breakdown:")
print(f"  {'Split':<8} {'Total':<8} {'Pos':<8} {'Neg':<8} {'CTs':<8} {'Ratio':<8}")
print(f"  {'-'*48}")
for s in ['train', 'val', 'test']:
    sub = df[df['split'] == s]
    if len(sub) == 0:
        continue
    p = (sub['class'] == 1).sum()
    ng = (sub['class'] == 0).sum()
    ct = sub['seriesuid'].nunique()
    r = p / max(1, ng)
    print(f"  {s:<8} {len(sub):<8} {p:<8} {ng:<8} {ct:<8} {r:<8.2f}")
print("=" * 60)

# Save statistics.csv
stats_rows = []
stats_rows.append({'Category': 'Overall', 'Metric': 'Total Samples', 'Value': total})
stats_rows.append({'Category': 'Overall', 'Metric': 'Unique CTs', 'Value': n_cts})
stats_rows.append({'Category': 'Overall', 'Metric': 'Positive', 'Value': n_pos})
stats_rows.append({'Category': 'Overall', 'Metric': 'Negative', 'Value': n_neg})
stats_rows.append({'Category': 'Overall', 'Metric': 'Pos/Neg Ratio', 'Value': round(n_pos/max(1,n_neg), 3)})

for label, name in [(0, 'Negative'), (1, 'Positive')]:
    sub = df[df['class'] == label]['hu_mean']
    stats_rows.append({'Category': f'HU {name}', 'Metric': 'mean', 'Value': round(sub.mean(), 1)})
    stats_rows.append({'Category': f'HU {name}', 'Metric': 'std', 'Value': round(sub.std(), 1)})
    stats_rows.append({'Category': f'HU {name}', 'Metric': 'min', 'Value': round(sub.min(), 1)})
    stats_rows.append({'Category': f'HU {name}', 'Metric': 'max', 'Value': round(sub.max(), 1)})

for s in ['train', 'val', 'test']:
    sub = df[df['split'] == s]
    n = len(sub)
    stats_rows.append({'Category': f'Split {s}', 'Metric': 'Total', 'Value': n})
    stats_rows.append({'Category': f'Split {s}', 'Metric': 'Positive', 'Value': (sub['class']==1).sum() if n>0 else 0})
    stats_rows.append({'Category': f'Split {s}', 'Metric': 'Negative', 'Value': (sub['class']==0).sum() if n>0 else 0})
    stats_rows.append({'Category': f'Split {s}', 'Metric': 'Unique CTs', 'Value': sub['seriesuid'].nunique() if n>0 else 0})

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(str(FIGURES_DIR / 'statistics.csv'), index=False)
print("\n  statistics.csv saved")

# Print first 100 rows of metadata
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 65)
print(f"\n=== First 100 rows of metadata ===")
print(df.head(100).to_string())

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
print(f"\n  Figures generated in paper_figs/:")
for f in sorted(FIGURES_DIR.glob('*.png')):
    print(f"    {f.name} ({f.stat().st_size} bytes)")
for f in sorted(FIGURES_DIR.glob('*.csv')):
    print(f"    {f.name}")
print(f"\n  data/processed/metadata.csv: {len(df)} rows")
print(f"  data/processed/patches/: {len(list(PATCHES_DIR.glob('*.npy')))} files")
