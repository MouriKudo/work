"""
LUNA16 数据下载脚本
====================
LUNA16 数据需要从官网注册下载: https://luna16.grand-challenge.org/Download/

使用方式 (在 bash 或 cmd 中执行):
1. 手动下载所有文件放到 data/raw/ 目录
2. 运行: python src/download_luna16.py --verify

或者使用 wget (需要先在浏览器中登录获取 cookie):
    python src/download_luna16.py --download --cookie "your_session_cookie"
"""

import argparse
import os
import hashlib
import csv
import sys
import io
# Fix encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
from datetime import datetime
from pathlib import Path

# LUNA16 文件的已知大小（字节）—— 用于初步校验
EXPECTED_SIZES = {
    "subset0.zip": 10566451200,
    "subset1.zip": 10456453120,
    "subset2.zip": 10509639680,
    "subset3.zip": 10559150080,
    "subset4.zip": 10554439680,
    "subset5.zip": 10535956480,
    "subset6.zip": 10535454720,
    "subset7.zip": 10539755520,
    "subset8.zip": 10504898560,
    "subset9.zip": 10511861760,
    "annotations.csv": 54000,
    "candidates.csv": 5500000,
    "sampleSubmission.csv": 2100000,
}

# LUNA16 官方 MD5（如有提供）
OFFICIAL_MD5 = {
    # 示例: "subset0.zip": "abc123..."
    # 官方通常不提供 MD5，这里留空
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def compute_md5(filepath):
    """计算文件的 MD5 哈希"""
    print(f"  Computing MD5 for {os.path.basename(filepath)}...")
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_file_size(filepath):
    """获取文件大小（字节）"""
    return os.path.getsize(filepath)


def generate_manifest():
    """
    遍历 data/raw/ 下所有文件，生成 data_manifest.csv
    包含字段: filename, size_bytes, size_mb, md5, expected_size_match
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "data_manifest.csv"

    rows = []
    total_size = 0

    for filepath in sorted(RAW_DIR.iterdir()):
        if filepath.is_file() and filepath.suffix in [".zip", ".csv"]:
            size = compute_file_size(filepath)
            md5 = compute_md5(filepath)
            total_size += size

            expected = EXPECTED_SIZES.get(filepath.name, None)
            size_match = "OK" if expected is None else ("OK" if abs(size - expected) / expected < 0.05 else "MISMATCH")

            rows.append({
                "filename": filepath.name,
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
                "md5": md5,
                "expected_size_match": size_match,
            })

            print(f"  {filepath.name}: {size / (1024*1024):.1f} MB, MD5={md5[:16]}..., size={size_match}")

    # 写入 CSV
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "size_bytes", "size_mb", "md5", "expected_size_match"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Manifest saved to {manifest_path}")
    print(f"  Total: {len(rows)} files, {total_size / (1024*1024*1024):.2f} GB")

    return rows


def check_all_present():
    """检查所有 12 个必需文件是否存在"""
    required = list(EXPECTED_SIZES.keys())
    present = []
    missing = []

    for name in required:
        path = RAW_DIR / name
        if path.exists():
            size = os.path.getsize(path)
            expected = EXPECTED_SIZES[name]
            if hasattr(expected, '__iter__'):
                # annotations.csv 之类大小可变
                present.append(f"  ✓ {name} ({size / 1024:.1f} KB)")
            elif abs(size - expected) / expected < 0.3:
                present.append(f"  ✓ {name} ({size / (1024**3):.1f} GB)")
            else:
                present.append(f"  ⚠ {name} (size={size} vs expected={expected}, possible corruption)")
        else:
            missing.append(f"  ✗ {name}")

    print("=" * 60)
    print("LUNA16 数据完整性检查")
    print("=" * 60)
    for line in present:
        print(line)
    if missing:
        print("\n缺失文件:")
        for line in missing:
            print(line)
        print(f"\n请从 https://luna16.grand-challenge.org/Download/ 下载")
    else:
        print(f"\n✓ 所有 {len(required)} 个文件齐全")
    print("=" * 60)


def print_data_structure():
    """打印预期的数据目录结构"""
    print("""
data/raw/ 目录应包含以下文件:
================================
data/raw/
├── subset0.zip      (~10 GB)  CT scans, subset 0
├── subset1.zip      (~10 GB)  CT scans, subset 1
├── subset2.zip      (~10 GB)  CT scans, subset 2
├── subset3.zip      (~10 GB)  CT scans, subset 3
├── subset4.zip      (~10 GB)  CT scans, subset 4
├── subset5.zip      (~10 GB)  CT scans, subset 5
├── subset6.zip      (~10 GB)  CT scans, subset 6
├── subset7.zip      (~10 GB)  CT scans, subset 7
├── subset8.zip      (~10 GB)  CT scans, subset 8
├── subset9.zip      (~10 GB)  CT scans, subset 9
├── annotations.csv  (~50 KB)  结节标注 (系列UID, 坐标, 直径)
├── candidates.csv   (~5 MB)   候选位置 + 概率
└── sampleSubmission.csv (~2 MB) Kaggle 提交样例

下载地址: https://luna16.grand-challenge.org/Download/

Download 页面可能需要 Cookie 认证。手动下载后，
将文件全部放入 data/raw/，然后运行:
    python src/download_luna16.py --verify
    python src/download_luna16.py --manifest
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LUNA16 Data Download Helper")
    parser.add_argument("--verify", action="store_true", help="检查所有文件是否齐全")
    parser.add_argument("--manifest", action="store_true", help="生成 data_manifest.csv")
    parser.add_argument("--info", action="store_true", help="打印数据结构说明")
    parser.add_argument("--all", action="store_true", help="执行全部检查 (verify + manifest)")

    args = parser.parse_args()

    if args.info or len(sys.argv) == 1:
        print_data_structure()

    if args.verify or args.all:
        check_all_present()

    if args.manifest or args.all:
        generate_manifest()
