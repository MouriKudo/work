import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path('D:/luna16-work')
PATCHES_DIR = PROJECT_ROOT / 'data' / 'processed' / 'patches'
METADATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'metadata.csv'
FIGURES_DIR = PROJECT_ROOT / 'paper_figs'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(METADATA_PATH))
print(f'Loaded {len(df)} rows')
print(f'Pos: {(df["class"]==1).sum()}, Neg: {(df["class"]==0).sum()}')

# ===== Sample Grid =====
pos_df = df[df['class'] == 1]
neg_df = df[df['class'] == 0]
n_pos_show = min(16, len(pos_df))
n_neg_show = min(16, len(neg_df))
pos_samples = pos_df.sample(n=n_pos_show, random_state=42) if n_pos_show > 0 else pos_df.iloc[:0]
neg_samples = neg_df.sample(n=n_neg_show, random_state=42) if n_neg_show > 0 else neg_df.iloc[:0]

n_cols = 8
n_rows = (n_pos_show + n_neg_show + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.5))
axes = axes.flatten()

for i, (_, row) in enumerate(pos_samples.iterrows()):
    patch = np.load(PATCHES_DIR / row['patch_file'])
    axes[i].imshow(patch[1], cmap='gray', vmin=0, vmax=1)
    axes[i].set_title(f'POS\n{row["seriesuid"][:8]}', fontsize=7)
    axes[i].axis('off')

for i, (_, row) in enumerate(neg_samples.iterrows()):
    idx = n_pos_show + i
    patch = np.load(PATCHES_DIR / row['patch_file'])
    axes[idx].imshow(patch[1], cmap='gray', vmin=0, vmax=1)
    axes[idx].set_title(f'NEG\n{row["seriesuid"][:8]}', fontsize=7, color='red')
    axes[idx].axis('off')

for j in range(n_pos_show + n_neg_show, len(axes)):
    axes[j].axis('off')

fig.suptitle(f'LUNA16 Patch Samples (Positive: {n_pos_show}, Negative: {n_neg_show})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / 'patch_samples_grid.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Sample grid saved')

# ===== HU Distribution =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pos_hu = df[df['class'] == 1]['hu_mean']
neg_hu = df[df['class'] == 0]['hu_mean']
bins = np.linspace(-1000, 500, 60)

axes[0].hist(pos_hu, bins=bins, alpha=0.7, color='red', label=f'Positive (n={len(pos_hu)})')
axes[0].hist(neg_hu, bins=bins, alpha=0.7, color='blue', label=f'Negative (n={len(neg_hu)})')
axes[0].set_xlabel('Mean HU')
axes[0].set_ylabel('Count')
axes[0].set_title('HU Distribution: Positive vs Negative Candidates')
axes[0].axvline(x=-600, color='gray', linestyle='--', alpha=0.5, label='Lung window center')
axes[0].axvline(x=-200, color='green', linestyle='--', alpha=0.5, label='Soft tissue')
axes[0].legend()

axes[1].hist(pos_hu, bins=bins, alpha=0.7, color='red', density=True, label='Positive')
axes[1].hist(neg_hu, bins=bins, alpha=0.7, color='blue', density=True, label='Negative')
axes[1].set_xlabel('Mean HU')
axes[1].set_ylabel('Density')
axes[1].set_title('Normalized HU Distribution')
axes[1].legend()
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / 'hu_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print('HU distribution saved')

# ===== Class Imbalance =====
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
counts = df['class'].value_counts().sort_index()

axes[0].bar(['Negative (0)', 'Positive (1)'], [counts.get(0, 0), counts.get(1, 0)],
            color=['blue', 'red'], edgecolor='black')
axes[0].set_ylabel('Count')
axes[0].set_title('Class Distribution')
for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
    axes[0].text(i, v + max(counts) * 0.02, str(v), ha='center', fontweight='bold')

per_uid = df.groupby('seriesuid').size()
axes[1].hist(per_uid, bins=30, color='green', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Candidates per CT')
axes[1].set_ylabel('Number of CTs')
axes[1].set_title(f'Candidates per Scan (median={per_uid.median():.0f})')

axes[2].pie([counts.get(0, 0), counts.get(1, 0)],
            labels=['Negative', 'Positive'],
            colors=['blue', 'red'],
            autopct='%1.1f%%',
            explode=(0, 0.05),
            startangle=90)
axes[2].set_title('Class Ratio')
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / 'class_imbalance.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Class imbalance saved')

print('All figures generated!')
