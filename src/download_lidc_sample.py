"""从 TCIA 官方入口下载 LIDC-IDRI XML 和小规模验证 series。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lidc_external import luna_seriesuids  # noqa: E402

XML_URL = (
    "https://wiki.cancerimagingarchive.net/download/attachments/1966254/"
    "LIDC-XML-only.zip?api=v2&modificationDate=1530215018015&version=1"
)
NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
MANIFEST_COLUMNS = [
    "artifact",
    "source_url",
    "local_path",
    "bytes",
    "md5",
    "patient_id",
    "seriesuid",
    "downloaded_at_utc",
]


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 MD5。"""

    digest = hashlib.md5()  # noqa: S324 - 数据完整性校验，不用于密码学
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def download_file(url: str, destination: Path, overwrite: bool = False) -> Path:
    """下载文件到目标路径；默认复用已存在且非空的文件。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "luna16-work/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def safe_extract_zip(archive: Path, destination: Path) -> int:
    """防止路径穿越后解压 ZIP，并返回成员数。"""

    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        bad_member = bundle.testzip()
        if bad_member:
            raise ValueError(f"ZIP CRC check failed: {bad_member}")
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != resolved_destination and resolved_destination not in target.parents:
                raise ValueError(f"unsafe ZIP member: {member.filename}")
        bundle.extractall(destination)
        return len(bundle.infolist())


def get_json(endpoint: str, parameters: dict[str, str]) -> list[dict]:
    """调用公开 NBIA Search REST API。"""

    query = urllib.parse.urlencode(parameters)
    url = f"{NBIA_BASE}/{endpoint}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "luna16-work/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def select_sample_series(
    series_records: list[dict],
    annotation_uids: set[str],
    luna_uids: set[str],
    max_bytes: int,
    requested_uid: str | None = None,
) -> dict:
    """选择带 XML、非 LUNA16 重叠且不超过容量限制的最小 CT series。"""

    eligible = []
    for record in series_records:
        uid = str(record.get("SeriesInstanceUID", ""))
        size = int(record.get("FileSize", 0) or 0)
        if requested_uid and uid != requested_uid:
            continue
        if (
            uid
            and uid in annotation_uids
            and uid not in luna_uids
            and str(record.get("Modality", "")) == "CT"
            and bool(record.get("AnnotationsFlag", False))
            and 0 < size <= max_bytes
        ):
            eligible.append(record)
    if not eligible:
        requested = f" uid={requested_uid}" if requested_uid else ""
        raise RuntimeError(
            f"no annotated non-overlap CT series satisfies max_bytes={max_bytes}{requested}"
        )
    return min(eligible, key=lambda record: int(record["FileSize"]))


def update_manifest(path: Path, entries: list[dict]) -> None:
    """按 artifact + seriesuid 更新下载校验清单。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
    indexed = {
        (row.get("artifact", ""), row.get("seriesuid", "")): row for row in existing
    }
    for entry in entries:
        indexed[(entry["artifact"], entry["seriesuid"])] = entry
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(
            indexed[key]
            for key in sorted(indexed, key=lambda item: (item[0], item[1]))
        )


def manifest_entry(
    artifact: str,
    source_url: str,
    local_path: Path,
    patient_id: str = "",
    seriesuid: str = "",
) -> dict:
    try:
        recorded_path = local_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        recorded_path = str(local_path.resolve())
    return {
        "artifact": artifact,
        "source_url": source_url,
        "local_path": recorded_path,
        "bytes": local_path.stat().st_size,
        "md5": file_md5(local_path),
        "patient_id": patient_id,
        "seriesuid": seriesuid,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载 LIDC-IDRI 官方 XML 和一个真实非重叠 CT series"
    )
    parser.add_argument("--download-xml", action="store_true")
    parser.add_argument("--download-sample", action="store_true")
    parser.add_argument("--seriesuid", help="指定 sample series；默认自动选最小合格项")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data/external/LIDC-IDRI",
    )
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=PROJECT_ROOT / "runs/external_validation/lidc_annotations.csv",
    )
    parser.add_argument(
        "--luna-raw",
        type=Path,
        default=PROJECT_ROOT / "data/raw",
    )
    parser.add_argument(
        "--luna-metadata",
        type=Path,
        default=PROJECT_ROOT / "data/processed/metadata.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "runs/external_validation/source_manifest.csv",
    )
    parser.add_argument(
        "--max-series-bytes",
        type=int,
        default=1024 * 1024 * 1024,
        help="单个 sample series 解压前 API 标称大小上限，默认 1 GiB",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.download_xml and not args.download_sample:
        raise SystemExit("至少指定 --download-xml 或 --download-sample")

    entries: list[dict] = []
    if args.download_xml:
        archive = download_file(
            XML_URL,
            args.data_root / "LIDC-XML-only.zip",
            overwrite=args.overwrite,
        )
        member_count = safe_extract_zip(archive, args.data_root / "xml")
        entries.append(manifest_entry("LIDC_XML_ONLY", XML_URL, archive))
        print(f"XML archive: {archive} ({member_count} ZIP members)")

    if args.download_sample:
        if not args.annotation_csv.exists():
            raise FileNotFoundError(
                f"run src/lidc_external.py after XML extraction first: {args.annotation_csv}"
            )
        annotations = pd.read_csv(args.annotation_csv, usecols=["seriesuid"])
        annotation_uids = set(annotations["seriesuid"].dropna().astype(str))
        luna_uids = luna_seriesuids(args.luna_raw, args.luna_metadata)
        series_records = get_json(
            "getSeries",
            {"Collection": "LIDC-IDRI", "Modality": "CT"},
        )
        selected = select_sample_series(
            series_records,
            annotation_uids,
            luna_uids,
            args.max_series_bytes,
            args.seriesuid,
        )
        uid = str(selected["SeriesInstanceUID"])
        patient_id = str(selected.get("PatientID", "unknown"))
        image_url = f"{NBIA_BASE}/getImage?{urllib.parse.urlencode({'SeriesInstanceUID': uid})}"
        series_dir = args.data_root / "dicom" / patient_id
        archive = download_file(
            image_url,
            series_dir / f"{uid}.zip",
            overwrite=args.overwrite,
        )
        member_count = safe_extract_zip(archive, series_dir / "series")
        entries.append(
            manifest_entry(
                "LIDC_SAMPLE_DICOM",
                image_url,
                archive,
                patient_id=patient_id,
                seriesuid=uid,
            )
        )
        print(
            f"Sample series: {patient_id} {uid} "
            f"({selected['ImageCount']} images, {member_count} ZIP members)"
        )

    update_manifest(args.manifest, entries)
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
