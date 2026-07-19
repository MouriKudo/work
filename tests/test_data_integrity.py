import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_manifest import build_manifest


class DataIntegrityTests(unittest.TestCase):
    def test_patient_level_splits_do_not_overlap(self):
        metadata = pd.read_csv(PROJECT_ROOT / "data/processed/metadata.csv")
        split_uids = {
            split: set(metadata.loc[metadata["split"] == split, "seriesuid"])
            for split in ("train", "val", "test")
        }
        self.assertFalse(split_uids["train"] & split_uids["val"])
        self.assertFalse(split_uids["train"] & split_uids["test"])
        self.assertFalse(split_uids["val"] & split_uids["test"])

    def test_processed_patch_shape_and_range(self):
        metadata = pd.read_csv(PROJECT_ROOT / "data/processed/metadata.csv")
        patch_path = PROJECT_ROOT / "data/processed/patches" / metadata.iloc[0]["patch_file"]
        patch = np.load(patch_path)
        self.assertEqual(patch.shape, (3, 64, 64))
        self.assertTrue(np.isfinite(patch).all())
        self.assertGreaterEqual(float(patch.min()), 0.0)
        self.assertLessEqual(float(patch.max()), 1.0)

    def test_manifest_marks_extracted_subset_without_fabricating_md5(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            subset = raw_dir / "subset0"
            subset.mkdir()
            (subset / "scan.mhd").write_text("ObjectType = Image", encoding="utf-8")
            manifest = build_manifest(raw_dir)
            row = manifest.loc[manifest["item"] == "subset0"].iloc[0]
            self.assertEqual(row["status"], "EXTRACTED_ONLY_ARCHIVE_MISSING")
            self.assertEqual(row["md5"], "")

            hashed = build_manifest(raw_dir, hash_extracted=True)
            hashed_row = hashed.loc[hashed["item"] == "subset0"].iloc[0]
            self.assertEqual(len(hashed_row["md5"]), 32)
            self.assertEqual(
                hashed_row["md5_scope"],
                "extracted_tree_paths_sizes_and_contents",
            )


if __name__ == "__main__":
    unittest.main()
