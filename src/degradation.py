"""面向 3×64×64 CT patch 的可参数化图像退化算子。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import get_worker_info

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEGRADATION_NAMES = (
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
    "window_shift",
    "jpeg",
    "resample",
)


def load_degradation_config(path: str | Path) -> dict:
    """读取并校验退化 YAML 配置。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    missing = set(DEGRADATION_NAMES) - set(config.get("levels", {}))
    if missing:
        raise ValueError(f"missing degradation definitions: {sorted(missing)}")
    for name in DEGRADATION_NAMES:
        if len(config["levels"][name]) != 5:
            raise ValueError(f"{name} must define exactly five levels")
    return config


def _as_numpy(patch: np.ndarray | torch.Tensor) -> tuple[np.ndarray, bool]:
    is_tensor = torch.is_tensor(patch)
    array = patch.detach().cpu().numpy() if is_tensor else np.asarray(patch)
    if array.ndim != 3:
        raise ValueError(f"expected C×H×W patch, got {array.shape}")
    return array.astype(np.float32, copy=True), is_tensor


def _restore_type(array: np.ndarray, original: np.ndarray | torch.Tensor, is_tensor: bool):
    array = np.ascontiguousarray(np.clip(array, 0.0, 1.0), dtype=np.float32)
    if not is_tensor:
        return array
    tensor = torch.from_numpy(array)
    return tensor.to(device=original.device, dtype=original.dtype)


def apply_degradation(
    patch: np.ndarray | torch.Tensor,
    name: str,
    parameters: Mapping[str, float | int],
    *,
    source_window: Mapping[str, float] | None = None,
    rng: np.random.Generator | None = None,
):
    """对单个 C×H×W patch 应用一种退化，并保持输入类型。

    Args:
        patch: `[0,1]` 的 NumPy 数组或 CPU/GPU Tensor。
        name: 六种退化名称之一。
        parameters: 对应算子的强度参数。
        source_window: `window_shift` 反归一化使用的原始 WW/WL。
        rng: 高斯噪声使用的随机数生成器。
    """
    array, is_tensor = _as_numpy(patch)
    rng = rng or np.random.default_rng()
    if name == "low_contrast":
        factor = float(parameters["factor"])
        center = array.mean(axis=(1, 2), keepdims=True)
        output = center + factor * (array - center)
    elif name == "gaussian_noise":
        sigma = float(parameters["sigma"])
        output = array + rng.normal(0.0, sigma, size=array.shape).astype(np.float32)
    elif name == "gaussian_blur":
        kernel_size = int(parameters["kernel_size"])
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Gaussian kernel_size must be a positive odd integer")
        sigma = float(parameters.get("sigma", 0.0))
        output = np.stack(
            [cv2.GaussianBlur(channel, (kernel_size, kernel_size), sigma) for channel in array]
        )
    elif name == "window_shift":
        source_window = source_window or {"width": 1500.0, "level": -600.0}
        old_width = float(source_window["width"])
        old_level = float(source_window["level"])
        new_width = old_width + float(parameters.get("ww_offset", 0.0))
        new_level = old_level + float(parameters.get("wl_offset", 0.0))
        if new_width <= 0:
            raise ValueError("shifted window width must be positive")
        hu = array * old_width + (old_level - old_width / 2.0)
        output = (hu - (new_level - new_width / 2.0)) / new_width
    elif name == "jpeg":
        quality = int(parameters["quality"])
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")
        channels = []
        for channel in array:
            source = np.round(np.clip(channel, 0, 1) * 255).astype(np.uint8)
            success, encoded = cv2.imencode(".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not success:
                raise RuntimeError("OpenCV failed to encode JPEG")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            channels.append(decoded.astype(np.float32) / 255.0)
        output = np.stack(channels)
    elif name == "resample":
        scale = float(parameters["scale"])
        if not 0 < scale <= 1:
            raise ValueError("resample scale must be in (0, 1]")
        height, width = array.shape[-2:]
        small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        output = np.stack(
            [
                cv2.resize(
                    cv2.resize(channel, small_size, interpolation=cv2.INTER_AREA),
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
                for channel in array
            ]
        )
    else:
        raise ValueError(f"unknown degradation: {name}")
    return _restore_type(output, patch, is_tensor)


class DegradationTransform:
    """固定退化类型和强度档位的 Dataset transform。"""

    def __init__(self, name: str, level: int, config: dict, seed: int = 0):
        if name not in DEGRADATION_NAMES:
            raise ValueError(name)
        if level not in range(1, 6):
            raise ValueError("level must be between 1 and 5")
        self.name = name
        self.level = level
        self.parameters = config["levels"][name][level - 1]
        self.source_window = config.get("source_window")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.worker_id: int | None = None

    def _worker_rng(self) -> np.random.Generator:
        """每个 DataLoader worker 使用独立且可复现的噪声序列。"""
        worker = get_worker_info()
        if worker is not None and self.worker_id != worker.id:
            self.worker_id = worker.id
            self.rng = np.random.default_rng((self.seed + worker.seed) % (2**32))
        return self.rng

    def __call__(self, patch):
        return apply_degradation(
            patch,
            self.name,
            self.parameters,
            source_window=self.source_window,
            rng=self._worker_rng(),
        )


class MixedDegradationTransform:
    """以给定概率混合干净数据与随机多类型退化，用于 Robust-PBIP。"""

    def __init__(
        self,
        config: dict,
        clean_probability: float | None = None,
        names: Sequence[str] | None = None,
        levels: Sequence[int] | None = None,
        seed: int = 0,
    ):
        training = config.get("robust_training", {})
        self.clean_probability = float(
            training.get("clean_probability", 0.35)
            if clean_probability is None else clean_probability
        )
        self.names = tuple(names or training.get("degradation_types", DEGRADATION_NAMES))
        self.levels = tuple(levels or training.get("allowed_levels", [1, 2, 3, 4]))
        if not 0 <= self.clean_probability <= 1:
            raise ValueError("clean_probability must be in [0,1]")
        if set(self.names) - set(DEGRADATION_NAMES):
            raise ValueError("unknown degradation in mixed transform")
        if any(level not in range(1, 6) for level in self.levels):
            raise ValueError("mixed degradation levels must be in [1,5]")
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.worker_id: int | None = None

    def _worker_rng(self) -> np.random.Generator:
        """避免多个 Windows DataLoader worker 复制完全相同的随机序列。"""
        worker = get_worker_info()
        if worker is not None and self.worker_id != worker.id:
            self.worker_id = worker.id
            self.rng = np.random.default_rng((self.seed + worker.seed) % (2**32))
        return self.rng

    def __call__(self, patch):
        rng = self._worker_rng()
        if rng.random() < self.clean_probability:
            return patch
        name = str(rng.choice(self.names))
        level = int(rng.choice(self.levels))
        return apply_degradation(
            patch,
            name,
            self.config["levels"][name][level - 1],
            source_window=self.config.get("source_window"),
            rng=rng,
        )


def save_demo_grid(
    patches: Sequence[np.ndarray],
    config: dict,
    level: int,
    output: Path,
    seed: int = 0,
) -> None:
    """生成 6×N 中心层退化样本网格。"""
    cell, label_width, top = 128, 150, 35
    rows, columns = len(DEGRADATION_NAMES), len(patches)
    canvas = Image.new("RGB", (label_width + columns * cell, top + rows * cell), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Six CT degradations, level={level}", fill="black", font=font)
    for row, name in enumerate(DEGRADATION_NAMES):
        draw.text((8, top + row * cell + cell // 2), name, fill="black", font=font)
        transform = DegradationTransform(name, level, config, seed + row)
        for column, patch in enumerate(patches):
            degraded = np.asarray(transform(patch))[1]
            image = Image.fromarray(np.round(degraded * 255).astype(np.uint8), mode="L")
            image = image.resize((cell, cell), Image.Resampling.NEAREST).convert("RGB")
            canvas.paste(image, (label_width + column * cell, top + row * cell))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成六类 CT 退化演示网格")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "src/configs/degradation.yaml")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/processed/metadata.csv")
    parser.add_argument("--patches", type=Path, default=PROJECT_ROOT / "data/processed/patches")
    parser.add_argument("--level", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper_figs/degradation_grid.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_degradation_config(args.config)
    metadata = pd.read_csv(args.metadata)
    test = metadata[metadata["split"] == "test"]
    positives = test[test["class"] == 1]
    negatives = test[test["class"] == 0]
    half = max(1, args.samples // 2)
    chosen = pd.concat(
        [
            positives.sample(min(half, len(positives)), random_state=args.seed),
            negatives.sample(min(args.samples - half, len(negatives)), random_state=args.seed),
        ]
    ).head(args.samples)
    patches = [np.load(args.patches / filename) for filename in chosen["patch_file"]]
    save_demo_grid(patches, config, args.level, args.output, args.seed)
    print(f"Saved {len(patches)} samples to {args.output}")


if __name__ == "__main__":
    main()
