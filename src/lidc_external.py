"""LIDC-IDRI DICOM/XML 解析、LUNA16 去重和外部 patch 提取工具。"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_COLUMNS = [
    "seriesuid",
    "reader_index",
    "nodule_id",
    "pixel_x",
    "pixel_y",
    "world_z",
    "sop_instance_uid",
    "roi_count",
    "contour_points",
    "xml_file",
]

EXTERNAL_METADATA_COLUMNS = [
    "seriesuid",
    "subset_id",
    "patch_file",
    "class",
    "world_x",
    "world_y",
    "world_z",
    "voxel_x",
    "voxel_y",
    "voxel_z",
    "reader_index",
    "nodule_id",
    "sop_instance_uid",
    "slice_alignment",
    "slice_z_error_mm",
    "source_xml",
    "split",
]


def local_name(tag: str) -> str:
    """移除 ElementTree 标签中的 XML namespace。"""

    return tag.rsplit("}", 1)[-1]


def child_text(element: ET.Element, names: Iterable[str]) -> str | None:
    """在元素后代中查找第一个匹配标签的非空文本。"""

    wanted = {name.lower() for name in names}
    for child in element.iter():
        if local_name(child.tag).lower() in wanted and child.text:
            return child.text.strip()
    return None


def _finite_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def parse_lidc_xml(path: str | Path) -> tuple[str | None, list[dict]]:
    """解析一个 LIDC XML，返回 SeriesInstanceUID 和医师级结节中心。

    只处理 ``unblindedReadNodule``（结节直径通常不小于 3 mm）。ROI 中
    ``inclusion=FALSE`` 的内部排除轮廓不参与中心计算。中心切片同时保留
    ``imageSOP_UID``，提取 patch 时优先使用 SOP UID 精确对齐；仅在 UID 缺失时
    才回退到最近的物理 Z 坐标。
    """

    path = Path(path)
    root = ET.parse(path).getroot()
    seriesuid = child_text(
        root,
        ("SeriesInstanceUid", "SeriesInstanceUID", "seriesInstanceUid"),
    )
    annotations: list[dict] = []
    reader_index = -1

    for element in root.iter():
        if local_name(element.tag) != "readingSession":
            continue
        reader_index += 1
        for nodule in element.iter():
            if local_name(nodule.tag) != "unblindedReadNodule":
                continue

            nodule_id = child_text(nodule, ("noduleID", "noduleId")) or "unknown"
            rois: list[dict] = []
            for roi in nodule.iter():
                if local_name(roi.tag) != "roi":
                    continue
                inclusion = child_text(roi, ("inclusion",))
                if inclusion and inclusion.strip().lower() in {"false", "0", "no"}:
                    continue

                world_z = _finite_float(
                    child_text(roi, ("imageZposition", "imageZPosition"))
                )
                sop_uid = child_text(
                    roi,
                    ("imageSOP_UID", "imageSOPUID", "imageSopUid"),
                )
                edge_points: list[tuple[float, float]] = []
                for edge in roi.iter():
                    if local_name(edge.tag) != "edgeMap":
                        continue
                    x = _finite_float(child_text(edge, ("xCoord",)))
                    y = _finite_float(child_text(edge, ("yCoord",)))
                    if x is not None and y is not None:
                        edge_points.append((x, y))
                if not edge_points or (world_z is None and not sop_uid):
                    continue
                rois.append(
                    {
                        "x": float(np.mean([point[0] for point in edge_points])),
                        "y": float(np.mean([point[1] for point in edge_points])),
                        "world_z": world_z,
                        "sop_instance_uid": sop_uid or "",
                        "points": len(edge_points),
                    }
                )

            if not rois:
                continue
            weighted_x = np.average(
                [roi["x"] for roi in rois], weights=[roi["points"] for roi in rois]
            )
            weighted_y = np.average(
                [roi["y"] for roi in rois], weights=[roi["points"] for roi in rois]
            )
            z_rois = [roi for roi in rois if roi["world_z"] is not None]
            if z_rois:
                weighted_z = float(
                    np.average(
                        [roi["world_z"] for roi in z_rois],
                        weights=[roi["points"] for roi in z_rois],
                    )
                )
                center_roi = min(
                    z_rois,
                    key=lambda roi: abs(float(roi["world_z"]) - weighted_z),
                )
            else:
                weighted_z = float("nan")
                center_roi = rois[len(rois) // 2]

            annotations.append(
                {
                    "seriesuid": seriesuid or "",
                    "reader_index": reader_index,
                    "nodule_id": nodule_id,
                    "pixel_x": float(weighted_x),
                    "pixel_y": float(weighted_y),
                    "world_z": weighted_z,
                    "sop_instance_uid": center_roi["sop_instance_uid"],
                    "roi_count": len(rois),
                    "contour_points": int(sum(roi["points"] for roi in rois)),
                    "xml_file": str(path),
                }
            )
    return seriesuid, annotations


def _xml_priority(path: Path) -> tuple[int, str]:
    """让官方更正/重新提交文件优先于归档中的旧版本。"""

    name = path.name.lower()
    is_correction = int("correction" in name or "resubmitted" in name)
    return is_correction, path.as_posix().lower()


def parse_xml_collection(
    xml_files: Sequence[str | Path],
) -> tuple[pd.DataFrame, list[dict]]:
    """解析 XML 集合，并按 SeriesInstanceUID 选择唯一有效版本。

    官方 XML-only 包含少量重复文件和一个更正文件。相同 series 只保留一个 XML；
    更正文件优先，其余重复项使用稳定的路径排序选择，避免重复生成 patch。
    """

    parsed_by_uid: dict[str, list[dict]] = defaultdict(list)
    records: list[dict] = []
    for raw_path in sorted((Path(path) for path in xml_files), key=lambda p: p.as_posix()):
        try:
            seriesuid, annotations = parse_lidc_xml(raw_path)
            record = {
                "xml_file": str(raw_path),
                "seriesuid": seriesuid or "",
                "annotations": len(annotations),
                "status": "PARSED",
                "selected_xml": "",
                "detail": "",
                "_annotations": annotations,
                "_path": raw_path,
            }
            records.append(record)
            if seriesuid:
                parsed_by_uid[seriesuid].append(record)
            else:
                record["status"] = "MISSING_SERIESUID"
        except Exception as error:  # 单个损坏 XML 不应中断全量清单
            records.append(
                {
                    "xml_file": str(raw_path),
                    "seriesuid": "",
                    "annotations": 0,
                    "status": "ERROR",
                    "selected_xml": "",
                    "detail": repr(error),
                    "_annotations": [],
                    "_path": raw_path,
                }
            )

    selected_annotations: list[dict] = []
    for group in parsed_by_uid.values():
        selected = sorted(
            group,
            key=lambda item: (-_xml_priority(item["_path"])[0], _xml_priority(item["_path"])[1]),
        )[0]
        selected_path = str(selected["_path"])
        for item in group:
            item["selected_xml"] = selected_path
            if item is selected:
                item["status"] = "OK"
                selected_annotations.extend(item["_annotations"])
            else:
                item["status"] = "DUPLICATE_SERIESUID"
                item["detail"] = f"selected={selected_path}"

    public_records = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    return pd.DataFrame(selected_annotations, columns=ANNOTATION_COLUMNS), public_records


def _looks_like_dicom(path: Path) -> bool:
    if path.suffix.lower() in {".dcm", ".dicom"}:
        return True
    if path.suffix:
        return False
    try:
        with path.open("rb") as stream:
            stream.seek(128)
            return stream.read(4) == b"DICM"
    except OSError:
        return False


def discover_dicom_series(root: str | Path) -> dict[str, list[str]]:
    """递归发现 DICOM 目录，并使用 GDCM 按 SeriesInstanceUID 分组。"""

    root = Path(root)
    if not root.exists():
        return {}
    directories = {
        path.parent
        for path in root.rglob("*")
        if path.is_file() and _looks_like_dicom(path)
    }
    discovered: dict[str, list[str]] = {}
    for directory in sorted(directories, key=lambda path: path.as_posix()):
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or []
        except RuntimeError:
            continue
        for seriesuid in series_ids:
            files = list(
                sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), seriesuid)
            )
            if files and str(seriesuid) not in discovered:
                discovered[str(seriesuid)] = files
    return discovered


def luna_seriesuids(
    raw_root: str | Path,
    metadata_path: str | Path | None = None,
) -> set[str]:
    """从 MHD 文件名和 metadata 双重收集 LUNA16 SeriesUID。"""

    raw_root = Path(raw_root)
    identifiers = (
        {path.stem for path in raw_root.rglob("*.mhd")} if raw_root.exists() else set()
    )
    if metadata_path and Path(metadata_path).exists():
        metadata = pd.read_csv(metadata_path, usecols=["seriesuid"])
        identifiers.update(metadata["seriesuid"].dropna().astype(str))
    return identifiers


def read_dicom_series(files: Sequence[str]) -> sitk.Image:
    """读取由 GDCM 排序后的 DICOM 文件列表。"""

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(list(files))
    return reader.Execute()


def read_dicom_series_with_sop(
    files: Sequence[str],
) -> tuple[sitk.Image, dict[str, int]]:
    """读取 DICOM，同时建立 SOPInstanceUID 到三维 Z 索引的映射。"""

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(list(files))
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    sop_to_index: dict[str, int] = {}
    for index in range(len(files)):
        if reader.HasMetaDataKey(index, "0008|0018"):
            uid = reader.GetMetaData(index, "0008|0018").strip().strip("\x00")
            if uid:
                sop_to_index[uid] = index
    return image, sop_to_index


def nearest_slice_index(image: sitk.Image, world_z: float) -> int:
    """根据每层物理坐标寻找最接近 XML imageZposition 的层。"""

    if not np.isfinite(float(world_z)):
        raise ValueError("world_z is not finite and SOP UID alignment was unavailable")
    z_coordinates = np.asarray(
        [
            image.TransformIndexToPhysicalPoint((0, 0, index))[2]
            for index in range(image.GetSize()[2])
        ]
    )
    return int(np.argmin(np.abs(z_coordinates - float(world_z))))


def resolve_slice_index(
    image: sitk.Image,
    world_z: float,
    sop_instance_uid: str,
    sop_to_index: dict[str, int],
) -> tuple[int, str, float]:
    """优先按 SOP UID 定位切片，并返回索引、方法和 Z 误差。"""

    sop_instance_uid = str(sop_instance_uid or "").strip()
    if sop_instance_uid and sop_instance_uid in sop_to_index:
        index = int(sop_to_index[sop_instance_uid])
        method = "SOP_UID"
    else:
        index = nearest_slice_index(image, float(world_z))
        method = "WORLD_Z_FALLBACK"
    actual_z = image.TransformIndexToPhysicalPoint((0, 0, index))[2]
    error = abs(float(actual_z) - float(world_z)) if np.isfinite(world_z) else float("nan")
    return index, method, error


def crop_patch_3x64(
    volume: np.ndarray,
    center_xyz: tuple[int, int, int],
) -> np.ndarray:
    """在边界用 -1000 HU 填充后裁剪 ``Z×Y×X = 3×64×64``。"""

    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape={volume.shape}")
    x, y, z = (int(value) for value in center_xyz)
    depth, height, width = volume.shape
    if not (0 <= x < width and 0 <= y < height and 0 <= z < depth):
        raise ValueError(
            f"center {(x, y, z)} is outside volume bounds {(width, height, depth)}"
        )
    padded = np.pad(
        volume,
        ((1, 1), (32, 32), (32, 32)),
        mode="constant",
        constant_values=-1000,
    )
    padded_z, padded_y, padded_x = z + 1, y + 32, x + 32
    patch = padded[
        padded_z - 1 : padded_z + 2,
        padded_y - 32 : padded_y + 32,
        padded_x - 32 : padded_x + 32,
    ]
    if patch.shape != (3, 64, 64):
        raise RuntimeError(f"unexpected patch shape: {patch.shape}")
    return patch


def lung_window(
    patch_hu: np.ndarray,
    width: float = 1500.0,
    level: float = -600.0,
) -> np.ndarray:
    """使用与 LUNA16 预处理一致的肺窗归一化。"""

    if width <= 0:
        raise ValueError("window width must be positive")
    lower = level - width / 2.0
    return np.clip(
        (patch_hu.astype(np.float32) - lower) / width,
        0.0,
        1.0,
    ).astype(np.float32)


def _safe_filename_component(value: object) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value))
    return cleaned.strip("._") or "unknown"


def extract_external_patches(
    annotations: pd.DataFrame,
    series: dict[str, list[str]],
    non_overlap: set[str],
    patches_dir: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    """为非 LUNA16 重叠、带 XML 阳性标注的 series 提取 patch。"""

    patches_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    records: list[dict] = []
    if annotations.empty:
        return pd.DataFrame(columns=EXTERNAL_METADATA_COLUMNS), records

    for seriesuid, group in annotations.groupby("seriesuid", sort=True):
        seriesuid = str(seriesuid)
        if seriesuid not in series:
            records.append(
                {
                    "seriesuid": seriesuid,
                    "status": "SKIPPED_MISSING_DICOM",
                    "requested_patches": len(group),
                    "extracted_patches": 0,
                    "failed_patches": 0,
                    "sop_aligned": 0,
                    "world_z_fallback": 0,
                    "detail": "",
                }
            )
            continue
        if seriesuid not in non_overlap:
            records.append(
                {
                    "seriesuid": seriesuid,
                    "status": "SKIPPED_LUNA16_OVERLAP",
                    "requested_patches": len(group),
                    "extracted_patches": 0,
                    "failed_patches": 0,
                    "sop_aligned": 0,
                    "world_z_fallback": 0,
                    "detail": "",
                }
            )
            continue

        extracted = failed = sop_aligned = fallback = 0
        errors: list[str] = []
        try:
            image, sop_to_index = read_dicom_series_with_sop(series[seriesuid])
            volume = sitk.GetArrayFromImage(image)
        except Exception as error:
            records.append(
                {
                    "seriesuid": seriesuid,
                    "status": "ERROR_READING_DICOM",
                    "requested_patches": len(group),
                    "extracted_patches": 0,
                    "failed_patches": len(group),
                    "sop_aligned": 0,
                    "world_z_fallback": 0,
                    "detail": repr(error),
                }
            )
            continue

        for annotation_index, row in group.reset_index(drop=True).iterrows():
            try:
                z_index, alignment, z_error = resolve_slice_index(
                    image,
                    float(row["world_z"]),
                    str(row.get("sop_instance_uid", "")),
                    sop_to_index,
                )
                x_index = int(round(float(row["pixel_x"])))
                y_index = int(round(float(row["pixel_y"])))
                patch = lung_window(
                    crop_patch_3x64(volume, (x_index, y_index, z_index))
                )
                nodule_component = _safe_filename_component(row["nodule_id"])
                filename = (
                    f"{seriesuid}_r{int(row['reader_index'])}_"
                    f"n{nodule_component}_{annotation_index:03d}_class1.npy"
                )
                np.save(patches_dir / filename, patch)
                world = image.TransformIndexToPhysicalPoint(
                    (x_index, y_index, z_index)
                )
                rows.append(
                    {
                        "seriesuid": seriesuid,
                        "subset_id": -1,
                        "patch_file": filename,
                        "class": 1,
                        "world_x": world[0],
                        "world_y": world[1],
                        "world_z": world[2],
                        "voxel_x": x_index,
                        "voxel_y": y_index,
                        "voxel_z": z_index,
                        "reader_index": int(row["reader_index"]),
                        "nodule_id": row["nodule_id"],
                        "sop_instance_uid": row.get("sop_instance_uid", ""),
                        "slice_alignment": alignment,
                        "slice_z_error_mm": z_error,
                        "source_xml": row.get("xml_file", ""),
                        "split": "external",
                    }
                )
                extracted += 1
                if alignment == "SOP_UID":
                    sop_aligned += 1
                else:
                    fallback += 1
            except Exception as error:  # 保留同一 series 中其他可用标注
                failed += 1
                errors.append(f"annotation={annotation_index}: {error!r}")

        status = "EXTRACTED" if failed == 0 else "PARTIAL"
        records.append(
            {
                "seriesuid": seriesuid,
                "status": status,
                "requested_patches": len(group),
                "extracted_patches": extracted,
                "failed_patches": failed,
                "sop_aligned": sop_aligned,
                "world_z_fallback": fallback,
                "detail": " | ".join(errors),
            }
        )
    return pd.DataFrame(rows, columns=EXTERNAL_METADATA_COLUMNS), records


def write_report(output: Path, stats: dict, lidc_root: Path) -> None:
    """写入可机器核查的 LIDC-IDRI 可行性与执行状态报告。"""

    status = stats["status"]
    text = f"""# LIDC-IDRI 外部验证数据处理报告

## 当前状态

- 状态：**{status}**
- 数据根目录：`{lidc_root}`
- XML 文件总数：{stats['xml_files']}
- XML 唯一 SeriesUID：{stats['xml_unique_series']}
- 去除的重复 XML：{stats['xml_duplicate_files']}
- 医师级结节标注：{stats['reader_nodule_annotations']}
- 发现 DICOM series：{stats['dicom_series']}
- 与 LUNA16 重叠 series：{stats['overlap_series']}
- 非重叠 DICOM series：{stats['non_overlap_series']}
- 已提取 patch：{stats['extracted_patches']}
- SOP UID 精确对齐 patch：{stats['sop_aligned_patches']}
- World-Z 回退对齐 patch：{stats['world_z_fallback_patches']}

## 方法约定

1. 使用 DICOM `SeriesInstanceUID` 与 LUNA16 `.mhd` 文件名/metadata 精确去重。
2. 同一 series 的重复 XML 仅选择一个版本；官方 correction/resubmitted 文件优先。
3. XML ROI 中心优先按 `imageSOP_UID` 映射到 DICOM 切片，缺失时才按物理 Z 最近邻回退。
4. 裁剪 3×64×64 patch，并使用 WW=1500、WL=-600 的肺窗归一化。
5. 当前 patch 保留医师级标注。若开展正式外部指标评估，必须另行冻结跨医师共识、
   负候选生成和匹配半径规则，不能直接把医师级重复标注当成独立病例。

## 完整外部验证前仍需满足

- 完整 DICOM 约 124–133 GB，需在容量足够的磁盘通过 TCIA Data Retriever 下载。
- 全量运行后抽检 DICOM 方向矩阵、SOP UID 对齐和 patch 中心。
- 在模型和阈值冻结后，按预注册的共识与负候选协议计算外部指标。
"""
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIDC-IDRI 外部数据解析与 patch 提取")
    parser.add_argument(
        "--lidc-root",
        type=Path,
        default=PROJECT_ROOT / "data/external/LIDC-IDRI",
        help="LIDC-IDRI 公共根目录",
    )
    parser.add_argument(
        "--dicom-root",
        type=Path,
        help="DICOM 根目录；默认与 --lidc-root 相同",
    )
    parser.add_argument(
        "--xml-root",
        type=Path,
        help="XML 根目录；默认与 --lidc-root 相同",
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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/external_validation",
    )
    parser.add_argument(
        "--patches-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/lidc_external_patches",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="对已下载且不与 LUNA16 重叠的阳性标注提取 patch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_root = args.xml_root or args.lidc_root
    dicom_root = args.dicom_root or args.lidc_root
    args.output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = list(xml_root.rglob("*.xml")) if xml_root.exists() else []
    annotation_frame, xml_records = parse_xml_collection(xml_files)
    annotation_frame.to_csv(args.output_dir / "lidc_annotations.csv", index=False)
    pd.DataFrame(
        xml_records,
        columns=[
            "xml_file",
            "seriesuid",
            "annotations",
            "status",
            "selected_xml",
            "detail",
        ],
    ).to_csv(args.output_dir / "xml_processing_log.csv", index=False)

    series = discover_dicom_series(dicom_root)
    luna_uids = luna_seriesuids(args.luna_raw, args.luna_metadata)
    overlap = set(series) & luna_uids
    non_overlap = set(series) - luna_uids
    annotation_uids = set(annotation_frame["seriesuid"].dropna().astype(str))
    case_frame = pd.DataFrame(
        [
            {
                "seriesuid": uid,
                "dicom_files": len(files),
                "has_xml_annotation": uid in annotation_uids,
                "overlaps_luna16": uid in overlap,
                "eligible_external": uid in non_overlap and uid in annotation_uids,
            }
            for uid, files in sorted(series.items())
        ],
        columns=[
            "seriesuid",
            "dicom_files",
            "has_xml_annotation",
            "overlaps_luna16",
            "eligible_external",
        ],
    )
    case_frame.to_csv(args.output_dir / "lidc_case_inventory.csv", index=False)
    case_frame[case_frame["eligible_external"]].to_csv(
        args.output_dir / "non_overlap_cases.csv",
        index=False,
    )

    if args.extract:
        metadata, extraction_records = extract_external_patches(
            annotation_frame,
            series,
            non_overlap,
            args.patches_dir,
        )
    else:
        metadata = pd.DataFrame(columns=EXTERNAL_METADATA_COLUMNS)
        extraction_records = [
            {
                "seriesuid": "",
                "status": "NOT_RUN",
                "requested_patches": 0,
                "extracted_patches": 0,
                "failed_patches": 0,
                "sop_aligned": 0,
                "world_z_fallback": 0,
                "detail": "pass --extract after installing DICOM data",
            }
        ]
    metadata.to_csv(args.output_dir / "lidc_external_metadata.csv", index=False)
    extraction_frame = pd.DataFrame(
        extraction_records,
        columns=[
            "seriesuid",
            "status",
            "requested_patches",
            "extracted_patches",
            "failed_patches",
            "sop_aligned",
            "world_z_fallback",
            "detail",
        ],
    )
    extraction_frame.to_csv(args.output_dir / "patch_processing_log.csv", index=False)

    duplicate_xml = sum(
        record["status"] == "DUPLICATE_SERIESUID" for record in xml_records
    )
    selected_xml = sum(record["status"] == "OK" for record in xml_records)
    extracted_patches = int(extraction_frame["extracted_patches"].fillna(0).sum())
    failed_patches = int(extraction_frame["failed_patches"].fillna(0).sum())
    if extracted_patches and failed_patches == 0:
        status = "SAMPLE_EXTRACTION_COMPLETE"
    elif extracted_patches:
        status = "SAMPLE_EXTRACTION_PARTIAL"
    elif series:
        status = "DICOM_READY_EXTRACTION_NOT_RUN"
    elif xml_files:
        status = "XML_PARSED_DICOM_NOT_PRESENT"
    else:
        status = "DATA_NOT_PRESENT"

    stats = {
        "status": status,
        "dicom_series": len(series),
        "xml_files": len(xml_files),
        "xml_unique_series": selected_xml,
        "xml_duplicate_files": duplicate_xml,
        "reader_nodule_annotations": len(annotation_frame),
        "luna_seriesuids": len(luna_uids),
        "overlap_series": len(overlap),
        "non_overlap_series": len(non_overlap),
        "eligible_annotated_series": int(case_frame["eligible_external"].sum()),
        "extracted_patches": extracted_patches,
        "failed_patches": failed_patches,
        "sop_aligned_patches": int(extraction_frame["sop_aligned"].fillna(0).sum()),
        "world_z_fallback_patches": int(
            extraction_frame["world_z_fallback"].fillna(0).sum()
        ),
    }
    (args.output_dir / "deduplication_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir / "feasibility_report.md", stats, args.lidc_root)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
