#!/usr/bin/env python
"""Generate training curve plot from CSV using pure PIL (no matplotlib needed)"""
import pandas as pd
import sys
from pathlib import Path
from PIL import Image, ImageDraw

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('D:/luna16-work/runs/resnet18_20260717_201929/training_curve.csv')
out_dir = csv_path.parent

df = pd.read_csv(csv_path)
epochs = df.index + 1

W, H = 1400, 700
margin_l, margin_r, margin_t, margin_b = 80, 50, 60, 50
cols, rows = 3, 2
sub_w, sub_h = (W - margin_l - margin_r - 60) // cols, (H - margin_t - margin_b - 50) // rows

img = Image.new('RGB', (W, H), (255, 255, 255))
draw = ImageDraw.Draw(img)
draw.text((W//2 - 220, 10), 'ResNet18 Baseline — LUNA16 20 Epochs (with Augmentation)', fill=(0,0,0))

plots = [
    ('Loss', ['train_loss', 'val_loss'], [(220,20,60), (30,144,255)]),
    ('Accuracy', ['train_acc', 'val_acc'], [(220,20,60), (30,144,255)]),
    ('AUC-ROC', ['val_auc'], [(34,139,34)]),
    ('F1 Score', ['val_f1'], [(255,140,0)]),
    ('AUC-ROC × Loss × F1 (Combined)', ['val_auc', 'val_f1', 'val_loss'], [(34,139,34), (255,140,0), (30,144,255)]),
    ('Train vs Val AUC (Gap Check)', ['val_auc'], [(128,0,128)]),
]

for pi, (title, cols_list, colors) in enumerate(plots):
    col = pi % cols; row = pi // cols
    x0 = margin_l + col * (sub_w + 22)
    y0 = margin_t + row * (sub_h + 25)

    draw.rectangle([x0, y0, x0+sub_w, y0+sub_h], outline=(220,220,220), width=1)
    draw.text((x0 + sub_w//2 - 30, y0 - 16), title, fill=(0,0,0))

    all_vals = []
    for cl in cols_list:
        all_vals.extend(df[cl].tolist())
    vmin, vmax = min(all_vals), max(all_vals)
    vrange = max(vmax - vmin, 1e-6)
    vmin -= vrange * 0.05; vmax += vrange * 0.05; vrange = vmax - vmin

    for ci, (cl, color) in enumerate(zip(cols_list, colors)):
        vals = df[cl].tolist()
        pts = []
        for i, v in enumerate(vals):
            px = x0 + 20 + int(i / max(len(epochs)-1, 1) * (sub_w - 40))
            py = int(y0 + sub_h - 15 - (v - vmin) / vrange * (sub_h - 30))
            pts.append((px, py))
        for a, b in zip(pts[:-1], pts[1:]):
            draw.line([a, b], fill=color, width=2)
        for p in pts:
            draw.ellipse([p[0]-2, p[1]-2, p[0]+2, p[1]+2], fill=color)
        # Legend
        ly = y0 + 8 + ci * 14
        draw.rectangle([x0+sub_w-180, ly, x0+sub_w-162, ly+10], fill=color)
        last_v = vals[-1]
        draw.text((x0+sub_w-157, ly-2), f'{cl}: {last_v:.4f}', fill=(0,0,0))

    draw.text((x0+20, y0+sub_h-12), f'Epoch 1', fill=(150,150,150))
    draw.text((x0+sub_w-55, y0+sub_h-12), f'Epoch {len(epochs)}', fill=(150,150,150))

# Summary text at bottom
summary = (
    f"Best Val AUC: {df['val_auc'].max():.4f} (epoch {df['val_auc'].idxmax()+1})  |  "
    f"Final: Val AUC={df['val_auc'].iloc[-1]:.4f}  Val F1={df['val_f1'].iloc[-1]:.4f}  Train Loss={df['train_loss'].iloc[-1]:.4f}"
)
draw.text((margin_l, H - 38), summary, fill=(80,80,80))

out_path = out_dir / 'training_curves.png'
img.save(str(out_path))
print(f'Saved: {out_path}')
print(f'Size: {W}x{H}, 6 subplots')
