"""LIDC-IDRI 外部验证的 DICOM/XML 解析、去重和 patch 对齐工具。"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def local_name(tag: str) -> str:
    """去除 XML namespace。"""
    return tag.rsplit("}", 1)[-1]


def child_text(element: ET.Element, names: Iterable[str]) -> str | None:
    wanted = {name.lower() for name in names}
    for child in element.iter():
        if local_name(child.tag).lower() in wanted and child.text:
            return child.text.strip()
    return None


def parse_lidc_xml(path: str | Path) -> tuple[str | None, list[dict]]:
    """解析单个 LIDC XML，输出 SeriesInstanceUID 与放射科医师结节中心。

    每个 `unblindedReadNodule` 聚合其全部 ROI 轮廓点；这是外部 patch 中心，
    不在此阶段合并不同阅片医师的结节，因为正式共识规则需单独声明。
    """
    path = Path(path)
    root = ET.parse(path).getroot()
    seriesuid = child_text(
        root,
        ("SeriesInstanceUid", "SeriesInstanceUID", "seriesInstanceUid"),
    )
    annotations = []
    reader_index = -1
    for element in root.iter():
        name = local_name(element.tag)
        if name == "readingSession":
            reader_index += 1
            for nodule in element.iter():
                if local_name(nodule.tag) != "unblindedReadNodule":
                    continue
                nodule_id = child_text(nodule, ("noduleID", "noduleId")) or "unknown"
                x_values, y_values, z_values = [], [], []
                roi_count = 0
                for roi in nodule.iter():
                    if local_name(roi.tag) != "roi":
                        continue
                    z_text = child_text(roi, ("imageZposition", "imageZPosition"))
                    edge_points = []
                    for edge in roi.iter():
                        if local_name(edge.tag) != "edgeMap":
                            continue
                        x_text = child_text(edge, ("xCoord",))
                        y_text = child_text(edge, ("yCoord",))
                        if x_text is not None and y_text is not None:
                            edge_points.append((float(x_text), float(y_text)))
                    if not edge_points:
                        continue
                    roi_count += 1
                    x_values.extend(point[0] for point in edge_points)
                    y_values.extend(point[1] for point in edge_points)
                    if z_text is not None:
                        z_values.extend([float(z_text)] * len(edge_points))
                if x_values and y_values and z_values:
                    annotations.append(
                        {
                            "seriesuid": seriesuid or "",
                            "reader_index": reader_index,
                            "nodule_id": nodule_id,
                            "pixel_x": float(np.mean(x_values)),
                            "pixel_y": float(np.mean(y_values)),
                            "world_z": float(np.mean(z_values)),
                            "roi_count": roi_count,
                            "contour_points": len(x_values),
                            "xml_file": str(path),
                        }
                    )
    return seriesuid, annotations


def discover_dicom_series(root: str | Path) -> dict[str, list[str]]:
    """递归发现 DICOM 目录并使用 GDCM 按 SeriesInstanceUID 分组。"""
    root = Path(root)
    if not root.exists():
        return {}
    directories = {path.parent for path in root.rglob("*") if path.is_file()}
    discovered: dict[str, list[str]] = {}
    for directory in sorted(directories):
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or []
        except RuntimeError:
            continue
        for seriesuid in series_ids:
            files = list(
                sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), seriesuid)
            )
            if files and seriesuid not in discovered:
                discovered[str(seriesuid)] = files
    return discovered


def luna_seriesuids(raw_root: str | Path, metadata_path: str | Path | None = None) -> set[str]:
    """从 MHD 文件名和现有 metadata 双重收集 LUNA16 SeriesUID。"""
    raw_root = Path(raw_root)
    identifiers = {path.stem for path in raw_root.rglob("*.mhd")} if raw_root.exists() else set()
    if metadata_path and Path(metadata_path).exists():
        identifiers.update(pd.read_csv(metadata_path)["seriesuid"].astype(str))
    return identifiers


def read_dicom_series(files: list[str]) -> sitk.Image:
    """按 GDCM 排序后的文件列表读取三维 CT。"""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    return reader.Execute()


def nearest_slice_index(image: sitk.Image, world_z: float) -> int:
    """根据每层物理坐标寻找最接近 XML imageZposition 的层。"""
    z_coordinates = np.asarray(
        [image.TransformIndexToPhysicalPoint((0, 0, index))[2] for index in range(image.GetSize()[2])]
    )
    return int(np.argmin(np.abs(z_coordinates - float(world_z))))


def crop_patch_3x64(volume: np.ndarray, center_xyz: tuple[int, int, int]) -> np.ndarray:
    """边界零填充后裁剪 Z×Y×X = 3×64×64。"""
    x, y, z = center_xyz
    padded = np.pad(volume, ((1, 1), (32, 32), (32, 32)), mode="constant", constant_values=-1000)
    z, y, x = z + 1, y + 32, x + 32
    return padded[z - 1 : z + 2, y - 32 : y + 32, x - 32 : x + 32]


def lung_window(patch_hu: np.ndarray, width: float = 1500.0, level: float = -600.0) -> np.ndarray:
    """与 LUNA16 预处理一致的肺窗归一化。"""
    lower = level - width / 2.0
    return np.clip((patch_hu.astype(np.float32) - lower) / width, 0.0, 1.0).astype(np.float32)


def extract_external_patches(
    annotations: pd.DataFrame,
    series: dict[str, list[str]],
    non_overlap: set[str],
    patches_dir: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    """仅对非重叠且存在 XML 阳性标注的序列提取对齐 patch。"""
    patches_dir.mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    for seriesuid, group in annotations.groupby("seriesuid"):
        if seriesuid not in non_overlap:
            records.append({"seriesuid": seriesuid, "status": "SKIPPED_OVERLAP_OR_MISSING_DICOM"})
            continue
        try:
            image = read_dicom_series(series[seriesuid])
            volume = sitk.GetArrayFromImage(image)
            for annotation_index, row in group.reset_index(drop=True).iterrows():
                z_index = nearest_slice_index(image, float(row["world_z"]))
                x_index = int(round(float(row["pixel_x"])))
                y_index = int(round(float(row["pixel_y"])))
                patch = lung_window(crop_patch_3x64(volume, (x_index, y_index, z_index)))
                filename = f"{seriesuid}_r{int(row['reader_index'])}_n{annotation_index:03d}_class1.npy"
                np.save(patches_dir / filename, patch)
                world = image.TransformIndexToPhysicalPoint((x_index, y_index, z_index))
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
                        "split": "external",
                    }
                )
            records.append({"seriesuid": seriesuid, "status": "EXTRACTED", "patches": len(group)})
        except Exception as error:  # 单个病例失败不应中断全量预处理
            records.append({"seriesuid": seriesuid, "status": "ERROR", "detail": repr(error)})
    return pd.DataFrame(rows), records


def write_report(
    output: Path,
    lidc_root: Path,
    series_count: int,
    xml_count: int,
    annotation_count: int,
    overlap_count: int,
    non_overlap_count: int,
) -> None:
    status = "READY_FOR_EXTRACTION" if series_count else "DATA_NOT_PRESENT"
    text = f"""# LIDC-IDRI 外部验证可行性报告

## 当前状态

- 状态：**{status}**
- 数据根目录：`{lidc_root}`
- 发现 DICOM series：{series_count}
- 解析 XML 文件：{xml_count}
- 放射科医师级结节标注：{annotation_count}
- 与 LUNA16 重叠 series：{overlap_count}
- 可作为非重叠外部病例的 series：{non_overlap_count}

## 方法约定

1. 使用 DICOM `SeriesInstanceUID` 与 LUNA16 `.mhd` 文件名精确去重。
2. XML 的 ROI 轮廓中心映射到最近物理 Z 层，再裁剪 3×64×64 patch。
3. 使用与 LUNA16 一致的肺窗（WW=1500、WL=-600）归一化。
4. 不同阅片医师的标注当前独立保留；正式报告前需声明共识合并规则。
5. XML 只直接提供结节标注。若需计算外部 AUC/F1/FROC，还必须定义负候选生成器、匹配半径及完整候选评估协议。

## 正式外部验证前检查清单

- [ ] 下载并校验完整 LIDC-IDRI DICOM 与 XML。
- [ ] 确认非重叠病例清单不含任何 LUNA16 series。
- [ ] 冻结阅片医师共识/恶性评分到二分类标签的规则。
- [ ] 生成不依赖测试标签的外部负候选。
- [ ] 抽检 DICOM 方向矩阵、XML 坐标和 patch 中心对齐。
- [ ] 冻结模型与阈值后一次性执行外部评估。
"""
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIDC-IDRI 外部验证可行性与预处理")
    parser.add_argument("--lidc-root", type=Path, default=PROJECT_ROOT / "data/external/LIDC-IDRI")
    parser.add_argument("--xml-root", type=Path, help="默认与 lidc-root 相同")
    parser.add_argument("--luna-raw", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--luna-metadata", type=Path, default=PROJECT_ROOT / "data/processed/metadata.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/external_validation")
    parser.add_argument("--patches-dir", type=Path, default=PROJECT_ROOT / "data/processed/lidc_external_patches")
    parser.add_argument("--extract", action="store_true", help="对非重叠阳性标注实际提取 patch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_root = args.xml_root or args.lidc_root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xml_files = list(xml_root.rglob("*.xml")) if xml_root.exists() else []
    annotations, xml_records = [], []
    for xml_file in xml_files:
        try:
            seriesuid, parsed = parse_lidc_xml(xml_file)
            annotations.extend(parsed)
            xml_records.append(
                {"xml_file": str(xml_file), "seriesuid": seriesuid or "", "annotations": len(parsed), "status": "OK"}
            )
        except Exception as error:
            xml_records.append(
                {"xml_file": str(xml_file), "seriesuid": "", "annotations": 0, "status": "ERROR", "detail": repr(error)}
            )
    annotation_frame = pd.DataFrame(annotations)
    if annotation_frame.empty:
        annotation_frame = pd.DataFrame(
            columns=["seriesuid", "reader_index", "nodule_id", "pixel_x", "pixel_y", "world_z", "roi_count", "contour_points", "xml_file"]
        )
    annotation_frame.to_csv(args.output_dir / "lidc_annotations.csv", index=False)
    pd.DataFrame(
        xml_records,
        columns=["xml_file", "seriesuid", "annotations", "status", "detail"],
    ).to_csv(args.output_dir / "xml_processing_log.csv", index=False)

    series = discover_dicom_series(args.lidc_root)
    luna_uids = luna_seriesuids(args.luna_raw, args.luna_metadata)
    overlap = set(series) & luna_uids
    non_overlap = set(series) - luna_uids
    case_frame = pd.DataFrame(
        [
            {
                "seriesuid": uid,
                "dicom_files": len(files),
                "has_xml_annotation": bool((annotation_frame["seriesuid"] == uid).any()),
                "overlaps_luna16": uid in overlap,
                "eligible_external": uid in non_overlap,
            }
            for uid, files in sorted(series.items())
        ],
        columns=["seriesuid", "dicom_files", "has_xml_annotation", "overlaps_luna16", "eligible_external"],
    )
    case_frame.to_csv(args.output_dir / "lidc_case_inventory.csv", index=False)
    case_frame[case_frame["eligible_external"]].to_csv(
        args.output_dir / "non_overlap_cases.csv", index=False
    )
    stats = {
        "status": "READY_FOR_EXTRACTION" if series else "DATA_NOT_PRESENT",
        "dicom_series": len(series),
        "xml_files": len(xml_files),
        "reader_nodule_annotations": len(annotation_frame),
        "luna_seriesuids": len(luna_uids),
        "overlap_series": len(overlap),
        "non_overlap_series": len(non_overlap),
    }
    (args.output_dir / "deduplication_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    extraction_records = []
    if args.extract:
        metadata, extraction_records = extract_external_patches(
            annotation_frame, series, non_overlap, args.patches_dir
        )
        metadata.to_csv(args.output_dir / "lidc_external_metadata.csv", index=False)
    else:
        extraction_records.append(
            {"status": "NOT_RUN", "detail": "pass --extract after installing LIDC-IDRI data"}
        )
    pd.DataFrame(extraction_records).to_csv(args.output_dir / "patch_processing_log.csv", index=False)
    write_report(
        args.output_dir / "feasibility_report.md",
        args.lidc_root,
        len(series),
        len(xml_files),
        len(annotation_frame),
        len(overlap),
        len(non_overlap),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
