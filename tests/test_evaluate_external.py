import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_external import choose_evaluation_source


class EvaluationSourceTests(unittest.TestCase):
    @staticmethod
    def _save_patch(directory: Path, name: str) -> None:
        np.save(directory / name, np.zeros((3, 64, 64), dtype=np.float32))

    def _build_luna(self, root: Path) -> tuple[Path, Path]:
        patches = root / "luna_patches"
        patches.mkdir()
        rows = []
        for index, label in enumerate([0, 1, 0, 1]):
            name = f"luna_{index}.npy"
            self._save_patch(patches, name)
            rows.append(
                {
                    "seriesuid": f"luna-{index // 2}",
                    "patch_file": name,
                    "class": label,
                    "split": "test",
                }
            )
        metadata = root / "luna.csv"
        pd.DataFrame(rows).to_csv(metadata, index=False)
        return metadata, patches

    def _write_dedup_stats(self, root: Path) -> Path:
        path = root / "dedup.json"
        path.write_text(
            json.dumps({"xml_unique_series": 2, "failed_patches": 0}),
            encoding="utf-8",
        )
        return path

    def test_positive_only_lidc_automatically_falls_back_to_luna(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            luna_metadata, luna_patches = self._build_luna(root)
            lidc_patches = root / "lidc_patches"
            lidc_patches.mkdir()
            self._save_patch(lidc_patches, "positive.npy")
            lidc_metadata = root / "lidc.csv"
            pd.DataFrame(
                [
                    {
                        "seriesuid": "lidc-one",
                        "patch_file": "positive.npy",
                        "class": 1,
                        "split": "external",
                    }
                ]
            ).to_csv(lidc_metadata, index=False)

            source, audit = choose_evaluation_source(
                "auto",
                lidc_metadata,
                lidc_patches,
                self._write_dedup_stats(root),
                luna_metadata,
                luna_patches,
            )
        self.assertEqual(source.name, "luna16_fixed_test")
        self.assertTrue(audit["fallback_used"])
        self.assertTrue(any("二分类" in reason for reason in audit["fallback_reasons"]))

    def test_valid_non_overlap_lidc_is_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            luna_metadata, luna_patches = self._build_luna(root)
            lidc_patches = root / "lidc_patches"
            lidc_patches.mkdir()
            rows = []
            for index, label in enumerate([0, 1]):
                name = f"external_{index}.npy"
                self._save_patch(lidc_patches, name)
                rows.append(
                    {
                        "seriesuid": f"lidc-{index}",
                        "patch_file": name,
                        "class": label,
                        "split": "external",
                    }
                )
            lidc_metadata = root / "lidc.csv"
            pd.DataFrame(rows).to_csv(lidc_metadata, index=False)

            source, audit = choose_evaluation_source(
                "auto",
                lidc_metadata,
                lidc_patches,
                self._write_dedup_stats(root),
                luna_metadata,
                luna_patches,
            )
        self.assertEqual(source.name, "lidc_idri_external")
        self.assertFalse(audit["fallback_used"])


if __name__ == "__main__":
    unittest.main()
