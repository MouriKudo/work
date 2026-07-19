import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metrics import binary_metrics, candidate_froc, find_best_f1_threshold


class MetricsTests(unittest.TestCase):
    def test_best_threshold_is_selected_from_scores(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        threshold, f1 = find_best_f1_threshold(labels, scores)
        self.assertAlmostEqual(threshold, 0.6)
        self.assertAlmostEqual(f1, 1.0)

    def test_binary_metrics_confusion_matrix(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.9, 0.2], 0.5)
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (1, 1, 1, 1))

    def test_candidate_froc_counts_false_positives_per_scan(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1])
        uids = np.array(["scan-a", "scan-a", "scan-b", "scan-b"])
        curve, points, cpm = candidate_froc(labels, scores, uids, [0.5])
        row = curve[np.isclose(curve["threshold"], 0.8)].iloc[0]
        self.assertAlmostEqual(row["fp_per_scan"], 0.5)
        self.assertAlmostEqual(row["sensitivity"], 0.5)
        self.assertGreaterEqual(cpm, 0.0)
        self.assertLessEqual(cpm, 1.0)
        self.assertEqual(len(points), 1)


if __name__ == "__main__":
    unittest.main()
