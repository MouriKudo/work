"""基于 backbone 特征的 LUNA16 Top-3 相似病例检索。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_utils import best_run_per_method, discover_main_runs, make_loader
from metrics import load_model


@torch.no_grad()
def extract_index(
    model: torch.nn.Module,
    loader,
    frame: pd.DataFrame,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """按 DataLoader 顺序提取 L2 归一化特征、标签和预测概率。"""
    model.eval()
    features, labels, probabilities = [], [], []
    for patches, target, _ in loader:
        patches = patches.to(device, non_blocking=True)
        output = model(patches)
        if isinstance(output, (tuple, list)):
            logits, batch_features = output[0], output[1]
        else:
            logits = output
            if not hasattr(model, "forward_features"):
                batch_features = model.backbone.avgpool(
                    model.backbone.layer4(
                        model.backbone.layer3(
                            model.backbone.layer2(
                                model.backbone.layer1(
                                    model.backbone.maxpool(
                                        model.backbone.relu(
                                            model.backbone.bn1(model.backbone.conv1(patches))
                                        )
                                    )
                                )
                            )
                        )
                    )
                ).flatten(1)
            else:
                batch_features = model.forward_features(patches)
        features.append(F.normalize(batch_features, dim=1).cpu().numpy().astype(np.float32))
        labels.append(target.numpy().astype(np.int64))
        probabilities.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
    feature_array = np.concatenate(features)
    label_array = np.concatenate(labels)
    probability_array = np.concatenate(probabilities)
    if len(frame) != len(feature_array):
        raise RuntimeError("metadata and extracted feature count differ")
    return {
        "features": feature_array,
        "labels": label_array,
        "probabilities": probability_array,
        # 固定 Unicode dtype，确保 NPZ 可在 allow_pickle=False 下安全加载。
        "seriesuids": frame["seriesuid"].astype(str).to_numpy(dtype=str),
        "patch_files": frame["patch_file"].astype(str).to_numpy(dtype=str),
    }


def save_index(index: dict[str, np.ndarray], path: Path) -> None:
    """将可移植特征索引保存为压缩 NPZ。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **index)


def load_index(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def retrieve_top_k(
    query_feature: np.ndarray,
    train_index: dict[str, np.ndarray],
    top_k: int = 3,
    exclude_seriesuid: str | None = None,
) -> pd.DataFrame:
    """使用余弦相似度返回训练集 Top-K 候选，可排除同一 CT。"""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query = np.asarray(query_feature, dtype=np.float32).reshape(-1)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    features = np.asarray(train_index["features"], dtype=np.float32)
    # 逐元素归约避免 Windows 上 NumPy BLAS 与并行训练进程的运行库冲突。
    similarity = np.sum(features * query[None, :], axis=1, dtype=np.float32)
    valid = np.ones(len(features), dtype=bool)
    if exclude_seriesuid:
        valid &= train_index["seriesuids"].astype(str) != str(exclude_seriesuid)
    candidate_indices = np.flatnonzero(valid)
    order = candidate_indices[np.argsort(similarity[candidate_indices])[::-1]][:top_k]
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "index": order,
            "seriesuid": train_index["seriesuids"][order],
            "patch_file": train_index["patch_files"][order],
            "label": train_index["labels"][order].astype(int),
            "cosine_similarity": similarity[order],
            "prediction_probability": train_index["probabilities"][order],
        }
    )


@torch.no_grad()
def extract_single_feature(
    model: torch.nn.Module, patch_path: Path, device: torch.device
) -> tuple[np.ndarray, float]:
    patch = torch.from_numpy(np.load(patch_path).copy()).float().unsqueeze(0).to(device)
    output = model(patch)
    if isinstance(output, (tuple, list)):
        logits, features = output[0], output[1]
    else:
        logits = output
        features = model.forward_features(patch)
    normalized = F.normalize(features, dim=1)[0].cpu().numpy().astype(np.float32)
    return normalized, float(torch.sigmoid(logits)[0].cpu())


def save_retrieval_figure(
    query_patch: Path,
    query_label: str,
    query_probability: float,
    results: pd.DataFrame,
    patches_dir: Path,
    output: Path,
) -> None:
    """展示查询 patch 及 Top-3 标签、相似度和预测概率。"""
    cell, top = 180, 55
    canvas = Image.new("RGB", (cell * (1 + len(results)), top + cell + 38), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    items = [(query_patch, f"Query y={query_label} p={query_probability:.3f}")]
    for row in results.itertuples():
        items.append(
            (
                patches_dir / row.patch_file,
                f"Top-{row.rank} y={row.label} sim={row.cosine_similarity:.3f} p={row.prediction_probability:.3f}",
            )
        )
    draw.text((10, 12), "Top-3 similar training cases (cosine feature retrieval)", fill="black", font=font)
    for index, (path, label) in enumerate(items):
        patch = np.load(path)
        image = Image.fromarray(np.round(np.clip(patch[1], 0, 1) * 255).astype(np.uint8), mode="L")
        image = image.resize((cell, cell), Image.Resampling.BILINEAR).convert("RGB")
        canvas.paste(image, (index * cell, top))
        draw.text((index * cell + 3, top + cell + 8), label, fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建特征索引并执行 Top-3 病例检索")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs/experiments_v2")
    parser.add_argument("--method", choices=["resnet18_augmented", "pbip_lite", "pbip_full"], default="pbip_full")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/processed/metadata.csv")
    parser.add_argument("--patches", type=Path, default=PROJECT_ROOT / "data/processed/patches")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/retrieval")
    parser.add_argument("--figure-dir", type=Path, default=PROJECT_ROOT / "paper_figs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--query-index", type=int, default=0, help="测试索引中的查询编号")
    parser.add_argument("--query-patch", type=Path, help="也可直接输入一个 .npy patch")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--allow-same-series", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.method}_train_index.npz"
    test_path = args.output_dir / f"{args.method}_test_index.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(args.metadata)
    must_build = args.rebuild or not (train_path.exists() and test_path.exists())
    model = None
    spec = None
    if must_build or args.query_patch:
        spec = best_run_per_method(discover_main_runs(args.runs_root))[args.method]
        model = load_model(spec.model_type, spec.checkpoint, spec.prototype_bank).to(device)
    if must_build:
        assert model is not None
        for split, path in (("train", train_path), ("test", test_path)):
            dataset, loader = make_loader(
                split, args.metadata, args.patches, args.batch_size, args.num_workers, device
            )
            index = extract_index(model, loader, dataset.df, device)
            save_index(index, path)
            print(f"Saved {split} index: {path} ({len(index['labels'])} samples)")
    train_index = load_index(train_path)
    test_index = load_index(test_path)
    if spec is None:
        spec = best_run_per_method(discover_main_runs(args.runs_root))[args.method]
    (args.output_dir / f"{args.method}_index_manifest.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "seed": spec.seed,
                "run_dir": str(spec.run_dir),
                "checkpoint": str(spec.checkpoint),
                "prototype_bank": str(spec.prototype_bank) if spec.prototype_bank else None,
                "train_samples": int(len(train_index["labels"])),
                "test_samples": int(len(test_index["labels"])),
                "feature_dimension": int(train_index["features"].shape[1]),
                "normalization": "L2",
                "similarity": "cosine",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.query_patch:
        assert model is not None
        query_path = args.query_patch.resolve()
        query_feature, query_probability = extract_single_feature(model, query_path, device)
        query_uid, query_label = None, "unknown"
    else:
        if not 0 <= args.query_index < len(test_index["labels"]):
            raise IndexError("query-index outside test index")
        query_path = args.patches / str(test_index["patch_files"][args.query_index])
        query_feature = test_index["features"][args.query_index]
        query_probability = float(test_index["probabilities"][args.query_index])
        query_uid = str(test_index["seriesuids"][args.query_index])
        query_label = str(int(test_index["labels"][args.query_index]))
    results = retrieve_top_k(
        query_feature,
        train_index,
        top_k=args.top_k,
        exclude_seriesuid=None if args.allow_same_series else query_uid,
    )
    results.insert(0, "query_patch", str(query_path))
    results.to_csv(args.output_dir / "retrieval_example.csv", index=False)
    save_retrieval_figure(
        query_path,
        query_label,
        query_probability,
        results,
        args.patches,
        args.figure_dir / "retrieval_top3_example.png",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
