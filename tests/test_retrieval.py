import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import retrieve_top_k


class RetrievalTest(unittest.TestCase):
    def test_cosine_order_and_uid_exclusion(self):
        index = {
            "features": np.asarray([[1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32),
            "seriesuids": np.asarray(["same", "other", "far"]),
            "patch_files": np.asarray(["a.npy", "b.npy", "c.npy"]),
            "labels": np.asarray([1, 1, 0]),
            "probabilities": np.asarray([0.9, 0.8, 0.1]),
        }
        result = retrieve_top_k(np.asarray([1, 0]), index, top_k=2, exclude_seriesuid="same")
        self.assertEqual(result.iloc[0]["seriesuid"], "other")
        self.assertEqual(result.iloc[1]["seriesuid"], "far")


if __name__ == "__main__":
    unittest.main()
