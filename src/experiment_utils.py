"""跨实验复用的模型加载、预测和指标工具。

该模块只组合现有 ``LUNA16Dataset``、``metrics.py`` 与模型定义，不重复实现
Dataset、ResNet18 或 PBIP-Lite。新增汇总、消融、鲁棒性、Grad-CAM 和检索脚本
均通过这里解析正式实验目录。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from luna16_dataset import LUNA16Dataset
from metrics import binary_metrics, collect_predictions, find_best_f1_threshold, load_model


METHOD_DIRS = {
    "resnet18_baseline": "resnet18_noaug",
    "resnet18_augmented": "resnet18_strong",
    "pbip_lite": "pbip_lite",
    "pbip_full": "pbip_contrast",
}

METHOD_LABELS = {
    "resnet18_baseline": "ResNet18 baseline",
    "resnet18_augmented": "ResNet18 augmented",
    "pbip_lite": "PBIP-Lite",
    "pbip_full": "PBIP-Lite full",
}


@dataclass(frozen=True)
class RunSpec:
    """一个可加载的正式实验。"""

    method: str
    seed: int
    run_dir: Path
    model_type: str
    checkpoint: Path
    prototype_bank: Path | None
    best_val_auc: float


def load_yaml(path: str | Path) -> dict:
    """安全读取 YAML 配置并确保根节点是映射。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")
    return config


def _resolve_config_path(value: str | Path, run_dir: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    candidate = (run_dir / path).resolve()
    if candidate.exists():
        return candidate
    project_candidate = (PROJECT_ROOT / path).resolve()
    return project_candidate


def run_spec_from_dir(method: str, seed: int, run_dir: Path) -> RunSpec:
    """从 ``config.json``/``results.json`` 恢复模型加载信息。"""
    run_dir = run_dir.resolve()
    result_path = run_dir / "results.json"
    config_path = run_dir / "config.json"
    checkpoint = run_dir / "best_model.pth"
    if not (result_path.exists() and config_path.exists() and checkpoint.exists()):
        raise FileNotFoundError(f"incomplete run directory: {run_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    is_pbip = method.startswith("pbip") or str(result.get("method", "")).startswith("pbip")
    bank = None
    if is_pbip:
        bank_value = config.get("prototype_bank") or result.get("args", {}).get("prototype_bank")
        if not bank_value:
            raise ValueError(f"PBIP run has no prototype_bank: {run_dir}")
        bank = _resolve_config_path(bank_value, run_dir)
        if not bank.exists():
            raise FileNotFoundError(f"prototype bank does not exist: {bank}")
    return RunSpec(
        method=method,
        seed=int(seed),
        run_dir=run_dir,
        model_type="pbip" if is_pbip else "resnet18",
        checkpoint=checkpoint,
        prototype_bank=bank,
        best_val_auc=float(result["best_val_auc"]),
    )


def discover_main_runs(
    root: str | Path = PROJECT_ROOT / "runs/experiments_v2",
) -> list[RunSpec]:
    """发现四种方法 × seed 0/1/2 的正式运行。"""
    root = Path(root)
    specs: list[RunSpec] = []
    for seed in (0, 1, 2):
        for method, dirname in METHOD_DIRS.items():
            specs.append(run_spec_from_dir(method, seed, root / f"seed_{seed}" / dirname))
    return specs


def best_run_per_method(specs: Iterable[RunSpec]) -> Dict[str, RunSpec]:
    """仅按验证 AUC 选择每种方法的代表模型。"""
    best: Dict[str, RunSpec] = {}
    for spec in specs:
        if spec.method not in best or spec.best_val_auc > best[spec.method].best_val_auc:
            best[spec.method] = spec
    return best


def make_loader(
    split: str,
    metadata: str | Path,
    patches: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    transform: Callable | None = None,
) -> Tuple[LUNA16Dataset, DataLoader]:
    dataset = LUNA16Dataset(metadata, patches, split=split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return dataset, loader


@torch.no_grad()
def predict_run(
    spec: RunSpec,
    split: str,
    metadata: str | Path = PROJECT_ROOT / "data/processed/metadata.csv",
    patches: str | Path = PROJECT_ROOT / "data/processed/patches",
    batch_size: int = 256,
    num_workers: int = 0,
    transform: Callable | None = None,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载一个 run 并返回 probability、label、seriesuid。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(spec.model_type, spec.checkpoint, spec.prototype_bank).to(device)
    _, loader = make_loader(
        split, metadata, patches, batch_size, num_workers, device, transform
    )
    return collect_predictions(model, loader, device)


def evaluate_run(
    spec: RunSpec,
    metadata: str | Path = PROJECT_ROOT / "data/processed/metadata.csv",
    patches: str | Path = PROJECT_ROOT / "data/processed/patches",
    batch_size: int = 256,
    num_workers: int = 0,
    test_transform: Callable | None = None,
    threshold: float | None = None,
    device: torch.device | None = None,
) -> dict:
    """统一评估 AUC、PR-AUC 和 F1。

    阈值默认只在干净验证集上选择；``test_transform`` 仅应用于测试集，适合
    退化鲁棒性评估，避免针对每个退化强度重新调阈值。
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(spec.model_type, spec.checkpoint, spec.prototype_bank).to(device)
    if threshold is None:
        _, val_loader = make_loader(
            "val", metadata, patches, batch_size, num_workers, device
        )
        val_prob, val_labels, _ = collect_predictions(model, val_loader, device)
        threshold, validation_f1 = find_best_f1_threshold(val_labels, val_prob)
    else:
        validation_f1 = None
    _, test_loader = make_loader(
        "test", metadata, patches, batch_size, num_workers, device, test_transform
    )
    test_prob, test_labels, test_uids = collect_predictions(model, test_loader, device)
    metrics = binary_metrics(test_labels, test_prob, float(threshold))
    return {
        "method": spec.method,
        "seed": spec.seed,
        "run_dir": str(spec.run_dir),
        "checkpoint": str(spec.checkpoint),
        "prototype_bank": str(spec.prototype_bank) if spec.prototype_bank else "",
        "best_val_auc": spec.best_val_auc,
        "threshold": float(threshold),
        "validation_f1": validation_f1,
        "auc": metrics["auc"],
        "pr_auc": metrics["average_precision"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "probabilities": test_prob,
        "labels": test_labels,
        "seriesuids": test_uids,
    }


def json_ready(record: dict) -> dict:
    """移除预测数组并把 NumPy 标量转换为 JSON 兼容类型。"""
    ignored = {"probabilities", "labels", "seriesuids"}
    result = {}
    for key, value in record.items():
        if key in ignored:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result
