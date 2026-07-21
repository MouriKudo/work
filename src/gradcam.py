"""ResNet18/PBIP-Lite 的 2.5D Grad-CAM 与四类典型样本可视化。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_utils import best_run_per_method, discover_main_runs, make_loader
from metrics import collect_predictions, find_best_f1_threshold, load_model


class GradCAM:
    """对二维卷积中间层生成类别 1（结节）的 Grad-CAM。

    三张相邻 CT 层作为输入通道，最终热图覆盖在中间层图像上，因此属于
    2.5D 可解释性可视化。
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._capture_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._capture_gradient)

    def _capture_activation(self, _module, _inputs, output):
        self.activations = output

    def _capture_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 2.5D `[B,H,W]` 或 3D `[B,D,H,W]` 热图与阳性概率。"""
        if inputs.ndim not in (4, 5):
            raise ValueError("Grad-CAM inputs must be BCHW or BCDHW")
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        # 使 full backward hook 按模块输入梯度路径触发，避免 PyTorch 的回退告警。
        inputs = inputs.detach().requires_grad_(True)
        output = self.model(inputs)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        logits.sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("target layer hooks did not capture tensors")
        spatial_dims = tuple(range(2, self.gradients.ndim))
        weights = self.gradients.mean(dim=spatial_dims, keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        mode = "trilinear" if inputs.ndim == 5 else "bilinear"
        cam = F.interpolate(cam, size=inputs.shape[2:], mode=mode, align_corners=False)
        cam = cam[:, 0]
        flattened = cam.flatten(1)
        broadcast_shape = (cam.shape[0],) + (1,) * (cam.ndim - 1)
        minima = flattened.min(dim=1).values.reshape(broadcast_shape)
        maxima = flattened.max(dim=1).values.reshape(broadcast_shape)
        cam = (cam - minima) / (maxima - minima).clamp_min(1e-8)
        return cam.detach().cpu(), torch.sigmoid(logits).detach().cpu()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()


def select_typical_samples(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    samples_per_class: int,
) -> pd.DataFrame:
    """按置信度自动筛选 TP/FP/FN/TN 典型候选。"""
    selected = frame.reset_index(drop=True).copy()
    selected["probability"] = probabilities
    selected["prediction"] = (probabilities >= threshold).astype(int)
    masks = {
        "TP": (selected["class"] == 1) & (selected["prediction"] == 1),
        "FP": (selected["class"] == 0) & (selected["prediction"] == 1),
        "FN": (selected["class"] == 1) & (selected["prediction"] == 0),
        "TN": (selected["class"] == 0) & (selected["prediction"] == 0),
    }
    groups = []
    for category, mask in masks.items():
        candidates = selected[mask].copy()
        ascending = category in {"FN", "TN"}
        candidates = candidates.sort_values("probability", ascending=ascending).head(samples_per_class)
        candidates["category"] = category
        candidates["threshold"] = threshold
        groups.append(candidates)
    return pd.concat(groups, ignore_index=True) if groups else selected.iloc[:0]


def overlay_cam(patch: np.ndarray, cam: np.ndarray) -> tuple[Image.Image, Image.Image]:
    """返回中心层灰度原图及其热力图叠加。"""
    center = np.clip(patch[1], 0, 1)
    gray = np.round(center * 255).astype(np.uint8)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    heat = cv2.applyColorMap(np.round(np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 0.55, heat, 0.45, 0)
    return Image.fromarray(rgb), Image.fromarray(overlay)


def save_summary_grid(
    selected: pd.DataFrame,
    patches_dir: Path,
    cams: np.ndarray,
    output: Path,
    method: str,
    samples_per_class: int,
) -> None:
    """将四类样本的原图和 CAM 两两并排保存。"""
    cell, label_width, top = 112, 70, 50
    width = label_width + samples_per_class * cell * 2
    height = top + 4 * cell
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 12), f"{method}: original / Grad-CAM", fill="black", font=font)
    category_rows = {category: index for index, category in enumerate(("TP", "FP", "FN", "TN"))}
    for selected_index, row in selected.reset_index(drop=True).iterrows():
        category = row["category"]
        row_index = category_rows[category]
        same_category = selected.iloc[: selected_index + 1]
        column_index = int((same_category["category"] == category).sum() - 1)
        patch = np.load(patches_dir / row["patch_file"])
        original, overlay = overlay_cam(patch, cams[selected_index])
        original = original.resize((cell, cell), Image.Resampling.BILINEAR)
        overlay = overlay.resize((cell, cell), Image.Resampling.BILINEAR)
        x = label_width + column_index * cell * 2
        y = top + row_index * cell
        canvas.paste(original, (x, y))
        canvas.paste(overlay, (x + cell, y))
        draw.text((x + 3, y + 3), f"p={row['probability']:.3f}", fill="white", font=font)
    for category, row_index in category_rows.items():
        draw.text((15, top + row_index * cell + cell // 2), category, fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 ResNet18/PBIP-Lite 四类 Grad-CAM")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs/experiments_v2")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/processed/metadata.csv")
    parser.add_argument("--patches", type=Path, default=PROJECT_ROOT / "data/processed/patches")
    parser.add_argument("--methods", nargs="+", default=["resnet18_augmented", "pbip_full"])
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/gradcam")
    parser.add_argument("--figure-dir", type=Path, default=PROJECT_ROOT / "paper_figs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("samples-per-class must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_runs = best_run_per_method(discover_main_runs(args.runs_root))
    test_frame = pd.read_csv(args.metadata)
    test_frame = test_frame[test_frame["split"] == "test"].reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    for method in args.methods:
        spec = selected_runs[method]
        model = load_model(spec.model_type, spec.checkpoint, spec.prototype_bank).to(device)
        _, val_loader = make_loader("val", args.metadata, args.patches, 256, 0, device)
        val_probability, val_labels, _ = collect_predictions(model, val_loader, device)
        threshold, _ = find_best_f1_threshold(val_labels, val_probability)
        _, test_loader = make_loader("test", args.metadata, args.patches, 256, 0, device)
        test_probability, _, _ = collect_predictions(model, test_loader, device)
        typical = select_typical_samples(
            test_frame, test_probability, threshold, args.samples_per_class
        )
        patch_batch = torch.stack(
            [
                torch.from_numpy(np.load(args.patches / filename).copy()).float()
                for filename in typical["patch_file"]
            ]
        ).to(device)
        target_layer = model.backbone.layer4[-1]
        with GradCAM(model, target_layer) as gradcam:
            cams, recomputed_probability = gradcam(patch_batch)
        typical["cam_probability"] = recomputed_probability.numpy()
        typical["method"] = method
        typical["seed"] = spec.seed
        typical["run_dir"] = str(spec.run_dir)
        typical.to_csv(args.output_dir / f"{method}_samples.csv", index=False)
        save_summary_grid(
            typical,
            args.patches,
            cams.numpy(),
            args.figure_dir / f"gradcam_{method}_four_categories.png",
            method,
            args.samples_per_class,
        )
        all_records.append(typical)
        print(f"Saved {len(typical)} Grad-CAM samples for {method}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.concat(all_records, ignore_index=True).to_csv(
        args.output_dir / "gradcam_samples.csv", index=False
    )


if __name__ == "__main__":
    main()
