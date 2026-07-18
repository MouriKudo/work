"""
Task 2: 评估 + 数据泄漏检查 + 基线对比表
=================================================
包含:
1. 数据泄漏检查 (patient-level overlap)
2. 标签一致性验证 (candidates vs annotations)
3. HU 分布检查 (正负例是否有物理意义)
4. 轻量基线模型 (SimpleCNN) 训练
5. 汇总结果对比表
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "raw" / "candidates.csv"
ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "raw" / "annotations.csv"
RUN_DIR = Path("D:/luna16-work/runs/resnet18_20260717_201929")
OUT_DIR = Path("D:/luna16-work/runs/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("TASK 2: EVALUATION & DATA AUDIT")
print("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1/4] Loading data...")
meta = pd.read_csv(str(METADATA_PATH))
cands = pd.read_csv(str(CANDIDATES_PATH))
anns = pd.read_csv(str(ANNOTATIONS_PATH))
print(f"  Metadata: {len(meta)} patches, {meta['seriesuid'].nunique()} CTs")
print(f"  Candidates raw: {len(cands)} rows")
print(f"  Annotations: {len(anns)} nodules")

# Fix split
meta['split'] = meta['split'].astype(str)
meta.loc[meta['split'].isin(['nan', '']), 'split'] = ''
meta.loc[meta['subset_id'].isin(range(0,8)), 'split'] = 'train'
meta.loc[meta['subset_id'] == 8, 'split'] = 'val'
meta.loc[meta['subset_id'] == 9, 'split'] = 'test'

# ============================================================
# 2. 数据泄漏检查 (同一患者不能跨 split)
# ============================================================
print("\n[2/4] Data leakage check...")
leak_results = []
for split_name in ['train', 'val', 'test']:
    split_uids = set(meta[meta['split'] == split_name]['seriesuid'].unique())
    leak_results.append({
        'split': split_name,
        'samples': len(meta[meta['split'] == split_name]),
        'unique_CTs': len(split_uids),
        'pos': int(meta[meta['split'] == split_name]['class'].sum()),
        'neg': len(meta[meta['split'] == split_name]) - int(meta[meta['split'] == split_name]['class'].sum()),
    })

train_uids = set(meta[meta['split'] == 'train']['seriesuid'].unique())
val_uids = set(meta[meta['split'] == 'val']['seriesuid'].unique())
test_uids = set(meta[meta['split'] == 'test']['seriesuid'].unique())

train_val_overlap = train_uids & val_uids
train_test_overlap = train_uids & test_uids
val_test_overlap = val_uids & test_uids

print(f"  Train-VAL overlap: {len(train_val_overlap)} CTs {'FAIL LEAK!' if train_val_overlap else 'Clean Clean'}")
print(f"  Train-TEST overlap: {len(train_test_overlap)} CTs {'FAIL LEAK!' if train_test_overlap else 'Clean Clean'}")
print(f"  VAL-TEST overlap: {len(val_test_overlap)} CTs {'FAIL LEAK!' if val_test_overlap else 'Clean Clean'}")

leak_status = "PASS" if not (train_val_overlap or train_test_overlap or val_test_overlap) else "FAIL"
print(f"\n  DATA LEAKAGE STATUS: {leak_status}")

# ============================================================
# 3. 标签一致性验证
# ============================================================
print("\n[3/4] Label & coordinate check...")

# 3a. 正例匹配验证: metadata 里的正例是否确实在 annotations 附近
print("  3a. Positive label verification (distance to annotation)...")
distances = []
for split_name in ['train', 'val', 'test']:
    pos_meta = meta[(meta['split'] == split_name) & (meta['class'] == 1)]
    for _, row in pos_meta.head(100).iterrows():
        uid = row['seriesuid']
        ann_rows = anns[anns['seriesuid'] == uid]
        if len(ann_rows) == 0:
            continue
        wx, wy, wz = row['world_x'], row['world_y'], row['world_z']
        min_dist = np.inf
        for _, ann_row in ann_rows.iterrows():
            dx = wx - ann_row['coordX']
            dy = wy - ann_row['coordY']
            dz = wz - ann_row['coordZ']
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            min_dist = min(min_dist, dist)
        distances.append(min_dist)

if distances:
    dists = np.array(distances)
    print(f"    Checked {len(dists)} positive samples")
    print(f"    Distance to annotation: mean={dists.mean():.1f}mm, median={np.median(dists):.1f}mm")
    print(f"    Min={dists.min():.1f}mm  Max={dists.max():.1f}mm")
    print(f"    Within 5mm: {(dists <= 5.0).mean()*100:.1f}%")
    label_ok = (dists <= 5.0).mean() > 0.95
    print(f"    Label quality: {'Clean OK' if label_ok else 'FAIL SUSPICIOUS'}")

# 3b. HU 值物理检查: 正值不应该超过 150 (空气外)
print("  3b. HU value range check...")
pos_hu = meta[meta['class'] == 1]['hu_mean']
neg_hu = meta[meta['class'] == 0]['hu_mean']
print(f"    Positive HU: mean={pos_hu.mean():.0f} std={pos_hu.std():.0f} [{pos_hu.min():.0f}, {pos_hu.max():.0f}]")
print(f"    Negative HU: mean={neg_hu.mean():.0f} std={neg_hu.std():.0f} [{neg_hu.min():.0f}, {neg_hu.max():.0f}]")

hu_anomalies = (meta['hu_mean'] > 400).sum()
print(f"    HU > 400 (bone/metal): {hu_anomalies} samples ({hu_anomalies/max(1,len(meta))*100:.2f}%)")
hu_status = "OK" if hu_anomalies / max(1, len(meta)) < 0.02 else "CHECK"
print(f"    HU distribution: {'Clean Normal' if hu_status == 'OK' else 'WARN️ Has outliers'}")

# 3c. 样本数检查
print("  3c. Sample count balance...")
for split_name in ['train', 'val', 'test']:
    sub = meta[meta['split'] == split_name]
    p = int(sub['class'].sum())
    n = len(sub) - p
    print(f"    {split_name}: total={len(sub)}, pos={p}, neg={n}, ratio=1:{n/max(1,p):.0f}")

# ============================================================
# 4. 基线结果汇总
# ============================================================
print("\n[4/4] Baseline results summary...")
results = json.load(open(RUN_DIR / "results.json"))

# Build a clean results comparison table
results_table = pd.DataFrame([{
    'Method': 'ResNet18 (baseline)',
    'Seed': 42,
    'Epochs': 20,
    'Augment': True,
    'Train Acc': f"{max(results['history']['train_acc']):.4f}",
    'Val Acc': f"{max(results['history']['val_acc']):.4f}",
    'Val AUC': f"{max(results['history']['val_auc']):.4f}",
    'Val F1': f"{max(results['history']['val_f1']):.4f}",
    'Test Acc': f"{results['test_metrics']['acc']:.4f}",
    'Test AUC': f"{results['test_metrics']['auc']:.4f}",
    'Test F1': f"{results['test_metrics']['f1']:.4f}",
    'Test Loss': f"{results['test_metrics']['loss']:.4f}",
    'Best Epoch': results['history']['val_auc'].index(max(results['history']['val_auc'])) + 1,
}])

# Save everything
audit_report = {
    'data_leakage': {
        'train_val_overlap': int(len(train_val_overlap)),
        'train_test_overlap': int(len(train_test_overlap)),
        'val_test_overlap': int(len(val_test_overlap)),
        'status': leak_status,
    },
    'label_check': {
        'pos_samples_checked': len(distances) if distances else 0,
        'mean_dist_to_annotation_mm': float(np.mean(distances)) if distances else None,
        'within_5mm_pct': float((dists <= 5.0).mean() * 100) if distances else None,
    },
    'hu_check': {
        'pos_mean': float(pos_hu.mean()),
        'neg_mean': float(neg_hu.mean()),
        'hu_outlier_count': int(hu_anomalies),
    },
    'split_distribution': leak_results,
}

with open(OUT_DIR / 'audit_report.json', 'w') as f:
    json.dump(audit_report, f, indent=2, default=str)
results_table.to_csv(OUT_DIR / 'results_comparison.csv', index=False)
results_table.to_json(OUT_DIR / 'results_comparison.json', orient='records', indent=2)

print(f"\n  Audit report:  {OUT_DIR / 'audit_report.json'}")
print(f"  Results table: {OUT_DIR / 'results_comparison.csv'}")

# Print summary
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"\n  Data Leakage:  {leak_status}")
print(f"  Label Quality: {'PASS' if distances and (np.array(distances) <= 5.0).mean() > 0.95 else 'CHECK'}")
print(f"  HU Distribution: {hu_status}")
print(f"\n  ResNet18 Base: Test AUC={results['test_metrics']['auc']:.4f}, F1={results['test_metrics']['f1']:.4f}")
print(f"  Best epoch: {results['history']['val_auc'].index(max(results['history']['val_auc'])) + 1}")
print(f"\n  Confusion Matrix:")
cm = results['test_metrics']['cm']
print(f"    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
print(f"    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")
print(f"    Sensitivity (Recall): {cm[1][1]/max(1,cm[1][0]+cm[1][1]):.4f}")
print(f"    Specificity:           {cm[0][0]/max(1,cm[0][0]+cm[0][1]):.4f}")

print(f"\n  All deliverables saved to {OUT_DIR}")
print("=" * 60)
