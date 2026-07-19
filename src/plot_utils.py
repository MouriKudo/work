"""Small dependency-light plotting helpers based on Pillow.

The project environment has shown intermittent native crashes in
``matplotlib.savefig``.  These helpers cover the simple line charts required by
the experiment pipeline without invoking a GUI or compiled plotting backend.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#17becf",
    "#8c564b",
)


def _finite_xy(x_values: Sequence[float], y_values: Sequence[float]):
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def render_line_chart(
    series: Mapping[str, Tuple[Sequence[float], Sequence[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    width: int = 700,
    height: int = 460,
) -> Image.Image:
    """Render a compact line chart and return a Pillow image."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, right, top, bottom = 72, 24, 42, 62
    plot_width = width - left - right
    plot_height = height - top - bottom

    cleaned = {}
    for label, (x_values, y_values) in series.items():
        x, y = _finite_xy(x_values, y_values)
        if x.size:
            cleaned[label] = (x, y)
    if not cleaned:
        draw.text((left, top), "No finite data", fill="black", font=font)
        return image

    if xlim is None:
        all_x = np.concatenate([values[0] for values in cleaned.values()])
        xmin, xmax = float(all_x.min()), float(all_x.max())
    else:
        xmin, xmax = (float(value) for value in xlim)
    if ylim is None:
        all_y = np.concatenate([values[1] for values in cleaned.values()])
        ymin, ymax = float(all_y.min()), float(all_y.max())
    else:
        ymin, ymax = (float(value) for value in ylim)
    if math.isclose(xmin, xmax):
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if math.isclose(ymin, ymax):
        ymin, ymax = ymin - 0.5, ymax + 0.5
    if ylim is None:
        padding = 0.05 * (ymax - ymin)
        ymin, ymax = ymin - padding, ymax + padding

    def point(x_value: float, y_value: float):
        px = left + (x_value - xmin) / (xmax - xmin) * plot_width
        py = top + (ymax - y_value) / (ymax - ymin) * plot_height
        return int(round(px)), int(round(py))

    for tick in range(6):
        fraction = tick / 5
        x_value = xmin + fraction * (xmax - xmin)
        y_value = ymin + fraction * (ymax - ymin)
        px = left + int(fraction * plot_width)
        py = top + plot_height - int(fraction * plot_height)
        draw.line((px, top, px, top + plot_height), fill="#eeeeee")
        draw.line((left, py, left + plot_width, py), fill="#eeeeee")
        draw.text((px - 12, top + plot_height + 8), f"{x_value:.2g}", fill="#333333", font=font)
        draw.text((8, py - 6), f"{y_value:.2g}", fill="#333333", font=font)

    draw.line((left, top, left, top + plot_height), fill="black", width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill="black",
        width=2,
    )
    draw.text((left, 14), title, fill="black", font=font)
    draw.text((left + plot_width // 2 - 30, height - 22), xlabel, fill="black", font=font)
    draw.text((8, 18), ylabel, fill="black", font=font)

    legend_x = left + 8
    legend_y = top + 8
    for index, (label, (x, y)) in enumerate(cleaned.items()):
        color = PALETTE[index % len(PALETTE)]
        points = [point(float(xv), float(yv)) for xv, yv in zip(x, y)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        for px, py in points:
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
        label_x = legend_x + (index % 3) * 185
        label_y = legend_y + (index // 3) * 16
        draw.line((label_x, label_y + 5, label_x + 18, label_y + 5), fill=color, width=3)
        draw.text((label_x + 24, label_y), label, fill="#222222", font=font)

    return image


def save_line_chart(
    series: Mapping[str, Tuple[Sequence[float], Sequence[float]]],
    output_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
) -> None:
    image = render_line_chart(series, title, xlabel, ylabel, xlim=xlim, ylim=ylim)
    image.save(output_path)
