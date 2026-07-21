"""由 YAML 驱动原型组件、K 值和 beta 消融训练。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_utils import load_yaml


def resolve_path(root: Path, value: str | Path) -> Path:
    """相对路径统一以项目根目录解析。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def stream_command(command: list[str], log_path: Path, dry_run: bool = False) -> None:
    """实时输出子进程日志，并将相同内容持久化到日志文件。"""
    printable = subprocess.list2cmdline(command)
    print(f"\n$ {printable}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(printable + "\n", encoding="utf-8")
        return
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def ensure_bank(
    key: str,
    config: dict,
    root: Path,
    python: Path,
    defaults: dict,
    logs_dir: Path,
    dry_run: bool,
) -> Path:
    """复用已有原型库，或基于缓存训练特征构建指定 K 的原型库。"""
    bank_cfg = config["prototype_banks"][key]
    if "reuse_bank" in bank_cfg:
        bank_path = resolve_path(root, bank_cfg["reuse_bank"])
        if not bank_path.exists():
            raise FileNotFoundError(bank_path)
        return bank_path

    output_dir = resolve_path(root, bank_cfg["output_dir"])
    bank_path = output_dir / "prototype_bank.pkl"
    if bank_path.exists():
        print(f"[skip] prototype bank exists: {bank_path}")
        return bank_path
    command = [
        str(python),
        "-u",
        str(root / "src/prototype_bank.py"),
        "--checkpoint",
        str(resolve_path(root, defaults["init_checkpoint"])),
        "--output_dir",
        str(output_dir),
        "--metadata",
        str(resolve_path(root, config["metadata"])),
        "--patches",
        str(resolve_path(root, config["patches"])),
        "--k",
        str(bank_cfg["k"]),
        "--n",
        str(bank_cfg.get("n", 20)),
        "--seed",
        str(defaults["seed"]),
        "--batch_size",
        str(defaults["batch_size"]),
        "--num_workers",
        str(defaults["num_workers"]),
    ]
    if bank_cfg.get("feature_file"):
        command.extend(["--feature_file", str(resolve_path(root, bank_cfg["feature_file"]))])
    stream_command(command, logs_dir / f"build_bank_{key}.log", dry_run)
    if not dry_run and not bank_path.exists():
        raise RuntimeError(f"prototype bank was not created: {bank_path}")
    return bank_path


def train_experiment(
    experiment: dict,
    config: dict,
    root: Path,
    python: Path,
    defaults: dict,
    logs_dir: Path,
    bank_path: Path,
    dry_run: bool,
) -> Path:
    """训练单个消融；完成的目录自动跳过。"""
    if experiment.get("reuse_run"):
        run_dir = resolve_path(root, experiment["reuse_run"])
        if not (run_dir / "results.json").exists():
            raise FileNotFoundError(f"reused run is incomplete: {run_dir}")
        print(f"[reuse] {experiment['name']}: {run_dir}")
        return run_dir

    run_dir = resolve_path(root, config["output_root"]) / experiment["name"]
    if (run_dir / "results.json").exists() and (run_dir / "best_model.pth").exists():
        print(f"[skip] completed: {run_dir}")
        return run_dir
    options = {**defaults, **{k: v for k, v in experiment.items() if k in defaults}}
    command = [
        str(python),
        "-u",
        str(root / "src/pbip_train.py"),
        "--epochs", str(options["epochs"]),
        "--batch_size", str(options["batch_size"]),
        "--alpha", str(options["alpha"]),
        "--beta", str(options["beta"]),
        "--prototype_temperature", str(options["prototype_temperature"]),
        "--top_k", str(options["top_k"]),
        "--lr", str(options["lr"]),
        "--dropout", str(options["dropout"]),
        "--augment", str(options["augment"]),
        "--seed", str(options["seed"]),
        "--num_workers", str(options["num_workers"]),
        "--metadata", str(resolve_path(root, config["metadata"])),
        "--patches", str(resolve_path(root, config["patches"])),
        "--prototype_bank", str(bank_path),
        "--init_checkpoint", str(resolve_path(root, options["init_checkpoint"])),
        "--output_dir", str(run_dir),
        "--amp" if options.get("amp", True) else "--no-amp",
    ]
    stream_command(command, logs_dir / f"{experiment['name']}.log", dry_run)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 PBIP 消融实验")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "src/configs/ablation.yaml")
    parser.add_argument("--only", nargs="*", help="只运行指定实验名称")
    parser.add_argument("--dry-run", action="store_true", help="只输出命令，不训练")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    root = resolve_path(args.config.resolve().parent, config.get("project_root", "../.."))
    python = resolve_path(root, config["python"])
    logs_dir = resolve_path(root, config["logs_dir"])
    defaults = config["defaults"]
    selected = [
        item for item in config["experiments"]
        if not args.only or item["name"] in set(args.only)
    ]
    unknown = set(args.only or []) - {item["name"] for item in selected}
    if unknown:
        raise ValueError(f"unknown experiments: {sorted(unknown)}")

    manifest = []
    bank_cache: dict[str, Path] = {}
    for experiment in selected:
        bank_key = experiment["bank"]
        if bank_key not in bank_cache:
            bank_cache[bank_key] = ensure_bank(
                bank_key, config, root, python, defaults, logs_dir, args.dry_run
            )
        run_dir = train_experiment(
            experiment, config, root, python, defaults, logs_dir,
            bank_cache[bank_key], args.dry_run,
        )
        manifest.append(
            {
                "name": experiment["name"],
                "group": experiment["group"],
                "bank": bank_key,
                "run_dir": str(run_dir),
                "reused": bool(experiment.get("reuse_run")),
            }
        )
    output_root = resolve_path(root, config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        (output_root / "ablation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Ablation jobs handled: {len(manifest)}")


if __name__ == "__main__":
    main()
