import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from download_lidc_sample import safe_extract_zip, select_sample_series


class LIDCDownloadTest(unittest.TestCase):
    def test_selects_smallest_annotated_non_overlap_ct(self):
        records = [
            {
                "SeriesInstanceUID": "overlap",
                "Modality": "CT",
                "AnnotationsFlag": True,
                "FileSize": 10,
            },
            {
                "SeriesInstanceUID": "large",
                "Modality": "CT",
                "AnnotationsFlag": True,
                "FileSize": 300,
            },
            {
                "SeriesInstanceUID": "small",
                "Modality": "CT",
                "AnnotationsFlag": True,
                "FileSize": 100,
            },
        ]
        selected = select_sample_series(
            records,
            annotation_uids={"overlap", "large", "small"},
            luna_uids={"overlap"},
            max_bytes=1000,
        )
        self.assertEqual(selected["SeriesInstanceUID"], "small")

    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.txt", "unsafe")
            with self.assertRaises(ValueError):
                safe_extract_zip(archive, root / "output")
            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
