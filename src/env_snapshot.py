"""Create reproducible environment and directory-structure deliverables."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import PIL
import SimpleITK
import sklearn
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def directory_lines(root: Path):
    lines = [f"{root}/"]
    for directory_name in ("data", "paper_code", "paper_figs", "runs", "src", "tests"):
        directory = root / directory_name
        if not directory.exists():
            continue
        direct_files = sorted(path.name for path in directory.iterdir() if path.is_file())
        direct_dirs = sorted(path.name for path in directory.iterdir() if path.is_dir())
        lines.append(f"  {directory_name}/")
        for name in direct_dirs[:20]:
            lines.append(f"    {name}/")
        for name in direct_files[:30]:
            lines.append(f"    {name}")
        omitted = max(0, len(direct_dirs) - 20) + max(0, len(direct_files) - 30)
        if omitted:
            lines.append(f"    ... ({omitted} additional entries)")
    for filename in ("README.md", "PROJECT_STATUS.md", "requirements.txt", ".gitignore"):
        if (root / filename).exists():
            lines.append(f"  {filename}")
    return lines


def render_terminal_snapshot(lines, output_path: Path) -> None:
    font = ImageFont.load_default()
    line_height = 16
    width = 1100
    height = 36 + line_height * len(lines)
    image = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 30), fill="#1f2937")
    draw.ellipse((12, 10, 22, 20), fill="#ef4444")
    draw.ellipse((30, 10, 40, 20), fill="#f59e0b")
    draw.ellipse((48, 10, 58, 20), fill="#10b981")
    draw.text((72, 8), "LUNA16 environment snapshot", fill="#f9fafb", font=font)
    for index, line in enumerate(lines):
        color = "#86efac" if index == 0 else "#e5e7eb"
        draw.text((14, 34 + index * line_height), line, fill=color, font=font)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "environment",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
    patches_dir = PROJECT_ROOT / "data" / "processed" / "patches"
    subsets = sorted((PROJECT_ROOT / "data" / "raw").glob("subset[0-9]"))
    report = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "SimpleITK": SimpleITK.__version__,
        "scikit_learn": sklearn.__version__,
        "opencv": cv2.__version__,
        "matplotlib": matplotlib.__version__,
        "Pillow": PIL.__version__,
        "raw_subsets": len(subsets),
        "processed_patches": len(list(patches_dir.glob("*.npy"))) if patches_dir.exists() else 0,
        "metadata_rows": len(metadata),
        "metadata_positive": int(metadata["class"].sum()) if not metadata.empty else 0,
        "metadata_negative": int((metadata["class"] == 0).sum()) if not metadata.empty else 0,
        "unique_cts": int(metadata["seriesuid"].nunique()) if not metadata.empty else 0,
    }
    environment_lines = [
        f"Python: {report['python']}",
        f"Platform: {report['platform']}",
        f"PyTorch: {report['torch']} | torchvision: {report['torchvision']}",
        f"CUDA: {report['cuda_runtime']} | GPU: {report['gpu']}",
        f"SimpleITK: {report['SimpleITK']} | OpenCV: {report['opencv']}",
        f"NumPy: {report['numpy']} | pandas: {report['pandas']}",
        f"scikit-learn: {report['scikit_learn']} | Pillow: {report['Pillow']}",
        "",
        f"Raw subsets: {report['raw_subsets']}",
        f"Processed patches: {report['processed_patches']}",
        f"Metadata: {report['metadata_rows']} rows, {report['unique_cts']} CTs",
        f"Class counts: positive={report['metadata_positive']}, negative={report['metadata_negative']}",
    ]
    structure = directory_lines(PROJECT_ROOT)
    full_lines = environment_lines + ["", "Project structure:"] + structure
    text = "\n".join(full_lines) + "\n"
    print(text, end="")
    (output_dir / "environment_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "environment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "directory_tree.txt").write_text(
        "\n".join(structure) + "\n", encoding="utf-8"
    )
    render_terminal_snapshot(full_lines, output_dir / "environment_snapshot.png")
    print(f"Saved environment deliverables to {output_dir}")


if __name__ == "__main__":
    main()
