"""
LUNA16 坐标转换工具
===================
world coordinates (mm) <-> voxel indices
LUNA16 的 annotations.csv 和 candidates.csv 使用世界坐标系（毫米），
需要根据每个 CT 的 .mhd 头文件中的 origin 和 spacing 做转换。

相关公式:
    voxel_x = round((world_x - origin_x) / spacing_x)
    voxel_y = round((world_y - origin_y) / spacing_y)
    voxel_z = round((world_z - origin_z) / spacing_z)
"""

import numpy as np
from typing import Tuple, Optional


def world_to_voxel(
    world_xyz: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """
    世界坐标 (mm) -> 体素索引 (整数, z, y, x 顺序)。

    Parameters
    ----------
    world_xyz : (x, y, z) in mm, LUNA16 坐标系
    origin    : (x, y, z) in mm, SimpleITK image.GetOrigin()
    spacing   : (x, y, z) in mm/pixel, SimpleITK image.GetSpacing()

    Returns
    -------
    voxel_indices : (x, y, z) as integers (rounded)

    Note
    ----
    LUNA16 的世界坐标是 (x, y, z) 顺序。
    SimpleITK 读取后的 numpy 数组 shape 是 (z, y, x)。
    """
    x = int(round((world_xyz[0] - origin[0]) / spacing[0]))
    y = int(round((world_xyz[1] - origin[1]) / spacing[1]))
    z = int(round((world_xyz[2] - origin[2]) / spacing[2]))
    return (x, y, z)


def voxel_to_world(
    voxel_xyz: Tuple[int, int, int],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    体素索引 -> 世界坐标 (mm)。

    Parameters
    ----------
    voxel_xyz : (x, y, z) integers
    origin    : (x, y, z) in mm
    spacing   : (x, y, z) in mm/pixel

    Returns
    -------
    world_xyz : (x, y, z) in mm
    """
    x = origin[0] + voxel_xyz[0] * spacing[0]
    y = origin[1] + voxel_xyz[1] * spacing[1]
    z = origin[2] + voxel_xyz[2] * spacing[2]
    return (x, y, z)


def world_to_voxel_safe(
    world_xyz: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
    image_shape: Tuple[int, int, int],
) -> Optional[Tuple[int, int, int]]:
    """
    世界坐标转体素索引（带边界检查）。

    Returns
    -------
    voxel_xyz or None if out of bounds
    """
    x, y, z = world_to_voxel(world_xyz, origin, spacing)
    max_z, max_y, max_x = image_shape
    if 0 <= x < max_x and 0 <= y < max_y and 0 <= z < max_z:
        return (x, y, z)
    return None


def extract_patch_bounds(
    center_xyz: Tuple[int, int, int],
    patch_size: int = 64,
    image_shape: Tuple[int, int, int] = None,
) -> Tuple[int, int, int, int, int, int]:
    """
    计算围绕中心体素的 patch 裁剪边界。

    Parameters
    ----------
    center_xyz   : (x, y, z) — 中心体素索引
    patch_size   : patch 的边长 (x, y 方向)
    image_shape  : (z, y, x) — 图像尺寸，用于边界裁剪

    Returns
    -------
    x_start, y_start, x_end, y_end, z_idx, patch_size_actual
    (z_idx 是中心层的 z 索引)
    """
    half = patch_size // 2
    x, y, z = center_xyz

    x_start = x - half
    y_start = y - half
    x_end = x + half
    y_end = y + half

    if image_shape is not None:
        max_z, max_y, max_x = image_shape
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(max_x, x_end)
        y_end = min(max_y, y_end)

    return (x_start, y_start, x_end, y_end, z, patch_size)


def match_candidate_to_annotation(
    candidate_coord: Tuple[float, float, float],
    annotations_df,
    seriesuid: str,
    tolerance_mm: float = 5.0,
) -> bool:
    """
    判断一个 candidate 是否匹配任何 annotation（正例 vs 负例）。

    Parameters
    ----------
    candidate_coord : (x, y, z) world coordinates
    annotations_df   : DataFrame with columns [seriesuid, coordX, coordY, coordZ, diameter_mm]
    seriesuid        : 当前 CT 的 series UID
    tolerance_mm     : 匹配容忍度（mm）

    Returns
    -------
    True if this candidate is a positive (matches an annotation)
    """
    anns = annotations_df[annotations_df["seriesuid"] == seriesuid]
    if len(anns) == 0:
        return False

    cx, cy, cz = candidate_coord
    for _, row in anns.iterrows():
        dx = cx - row["coordX"]
        dy = cy - row["coordY"]
        dz = cz - row["coordZ"]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dist <= tolerance_mm:
            return True
    return False


# ====================== 单元测试 ======================

def test_world_to_voxel():
    """基础测试: 原点对应 (0,0,0) 体素"""
    result = world_to_voxel(
        world_xyz=(0.0, 0.0, 0.0),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
    )
    assert result == (0, 0, 0), f"Expected (0,0,0), got {result}"


def test_world_to_voxel_offset():
    """偏移测试: origin 不为零"""
    result = world_to_voxel(
        world_xyz=(100.0, -50.0, 200.0),
        origin=(-100.0, 100.0, 0.0),
        spacing=(1.0, 2.0, 0.7),
    )
    # x: (100 - (-100)) / 1 = 200
    # y: (-50 - 100) / 2 = -75
    # z: (200 - 0) / 0.7 = 285.71.. -> 286
    expected = (200, -75, 286)
    assert result == expected, f"Expected {expected}, got {result}"


def test_roundtrip():
    """往返测试: world -> voxel -> world 应回到原点"""
    origin = (-200.0, -150.0, -500.0)
    spacing = (0.8, 0.8, 1.5)
    world = (-100.0, 50.0, 0.0)

    voxel = world_to_voxel(world, origin, spacing)
    world_back = voxel_to_world(voxel, origin, spacing)

    # 由于取整，可能有 0.5 * spacing 的误差
    for i in range(3):
        assert abs(world[i] - world_back[i]) <= spacing[i] * 0.6, \
            f"Mismatch at dim {i}: {world[i]} vs {world_back[i]}"


def test_match_candidate():
    """正负例匹配测试"""
    import pandas as pd
    ann_df = pd.DataFrame({
        "seriesuid": ["CT001", "CT001", "CT002"],
        "coordX": [10.0, 50.0, 0.0],
        "coordY": [20.0, 60.0, 0.0],
        "coordZ": [30.0, 70.0, 0.0],
        "diameter_mm": [5.0, 8.0, 3.0],
    })
    # 完全匹配
    assert match_candidate_to_annotation((10.0, 20.0, 30.0), ann_df, "CT001")
    # 接近匹配 (距离 < 5mm)
    assert match_candidate_to_annotation((12.0, 22.0, 30.0), ann_df, "CT001")
    # 不匹配 (距离 > 5mm)
    assert not match_candidate_to_annotation((100.0, 200.0, 300.0), ann_df, "CT001")
    # 不同 series
    assert not match_candidate_to_annotation((0.0, 0.0, 0.0), ann_df, "CT001")
    print("  All match tests passed!")


def test_patch_bounds():
    """patch 边界测试"""
    # 中心位置
    result = extract_patch_bounds((100, 100, 50), patch_size=64, image_shape=(200, 512, 512))
    assert result == (68, 68, 132, 132, 50, 64), f"Got {result}"

    # 边界裁剪
    result_edge = extract_patch_bounds((10, 10, 5), patch_size=64, image_shape=(200, 512, 512))
    assert result_edge[0] == 0 and result_edge[1] == 0, f"Should clip to 0, got {result_edge}"
    print("  All patch bounds tests passed!")


def run_all_tests():
    print("=" * 60)
    print("Running coordinate_utils unit tests...")
    print("=" * 60)
    test_world_to_voxel()
    print("  test_world_to_voxel: PASSED")
    test_world_to_voxel_offset()
    print("  test_world_to_voxel_offset: PASSED")
    test_roundtrip()
    print("  test_roundtrip: PASSED")
    test_patch_bounds()
    print("  test_patch_bounds: PASSED")
    test_match_candidate()
    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
