"""Generate a transparent manifest for LUNA16 archives, extracted subsets and CSVs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_summary(path: Path):
    files = [file for file in path.rglob("*") if file.is_file()]
    return len(files), sum(file.stat().st_size for file in files)


def md5_directory(path: Path) -> str:
    """Hash relative paths and full file contents in deterministic order."""
    digest = hashlib.md5()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(file.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_manifest(raw_dir: Path, hash_extracted: bool = False) -> pd.DataFrame:
    rows = []
    for subset_id in range(10):
        archive = raw_dir / f"subset{subset_id}.zip"
        extracted = raw_dir / f"subset{subset_id}"
        if archive.exists():
            rows.append(
                {
                    "item": f"subset{subset_id}",
                    "kind": "archive",
                    "path": archive.name,
                    "status": "ARCHIVE_PRESENT",
                    "file_count": 1,
                    "size_bytes": archive.stat().st_size,
                    "size_gb": round(archive.stat().st_size / 1024**3, 4),
                    "md5": md5_file(archive),
                    "md5_scope": "downloaded_zip",
                }
            )
        elif extracted.exists():
            file_count, size_bytes = directory_summary(extracted)
            extracted_md5 = md5_directory(extracted) if hash_extracted else ""
            rows.append(
                {
                    "item": f"subset{subset_id}",
                    "kind": "extracted_directory",
                    "path": extracted.name,
                    "status": "EXTRACTED_ONLY_ARCHIVE_MISSING",
                    "file_count": file_count,
                    "size_bytes": size_bytes,
                    "size_gb": round(size_bytes / 1024**3, 4),
                    "md5": extracted_md5,
                    "md5_scope": (
                        "extracted_tree_paths_sizes_and_contents"
                        if hash_extracted
                        else "unavailable_without_downloaded_zip"
                    ),
                }
            )
        else:
            rows.append(
                {
                    "item": f"subset{subset_id}",
                    "kind": "missing",
                    "path": "",
                    "status": "MISSING",
                    "file_count": 0,
                    "size_bytes": 0,
                    "size_gb": 0.0,
                    "md5": "",
                    "md5_scope": "not_available",
                }
            )

    for filename in ("annotations.csv", "candidates.csv", "sampleSubmission.csv"):
        path = raw_dir / filename
        if path.exists():
            rows.append(
                {
                    "item": path.stem,
                    "kind": "csv",
                    "path": filename,
                    "status": "PRESENT",
                    "file_count": 1,
                    "size_bytes": path.stat().st_size,
                    "size_gb": round(path.stat().st_size / 1024**3, 6),
                    "md5": md5_file(path),
                    "md5_scope": "file",
                }
            )
        else:
            rows.append(
                {
                    "item": path.stem,
                    "kind": "csv",
                    "path": filename,
                    "status": "MISSING",
                    "file_count": 0,
                    "size_bytes": 0,
                    "size_gb": 0.0,
                    "md5": "",
                    "md5_scope": "not_available",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the LUNA16 data manifest")
    parser.add_argument("--raw_dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/raw/data_manifest.csv")
    parser.add_argument(
        "--hash_extracted",
        action="store_true",
        help="Read all extracted CT data and compute deterministic tree-content MD5 values",
    )
    args = parser.parse_args()
    manifest = build_manifest(args.raw_dir.resolve(), hash_extracted=args.hash_extracted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(manifest.to_string(index=False))
    print(f"Saved {len(manifest)} manifest rows to {args.output}")


if __name__ == "__main__":
    main()
