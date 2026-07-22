import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stats_test import (
    bootstrap_grouped_predictions,
    calculate_bootstrap_ci,
    calculate_multiseed_summary,
    run_significance_tests,
)


def make_predictions(method: str = "method_a", seed: int = 0) -> pd.DataFrame:
    """构造每个 CT 均同时包含正负类的可重复测试数据。"""

    rows = []
    for index in range(4):
        rows.extend(
            [
                {
                    "dataset": "synthetic",
                    "method": method,
                    "seed": seed,
                    "seriesuid": f"scan-{index}",
                    "label": 0,
                    "probability": 0.1 + 0.03 * index,
                    "threshold": 0.5,
                    "prediction": 0,
                },
                {
                    "dataset": "synthetic",
                    "method": method,
                    "seed": seed,
                    "seriesuid": f"scan-{index}",
                    "label": 1,
                    "probability": 0.9 - 0.03 * index,
                    "threshold": 0.5,
                    "prediction": 1,
                },
            ]
        )
    return pd.DataFrame(rows)


class BootstrapTests(unittest.TestCase):
    def test_cluster_bootstrap_is_reproducible(self):
        predictions = make_predictions()
        first = calculate_bootstrap_ci(
            predictions, n_bootstrap=100, random_seed=7
        )
        second = calculate_bootstrap_ci(
            predictions, n_bootstrap=100, random_seed=7
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["metric"]), {"accuracy", "precision", "recall", "f1", "auc"})
        self.assertTrue((first["bootstrap_valid"] == 100).all())
        self.assertTrue((first["resample_unit"] == "seriesuid").all())

    def test_grouped_bootstrap_keeps_method_and_seed(self):
        predictions = pd.concat(
            [make_predictions("method_a", 0), make_predictions("method_b", 1)],
            ignore_index=True,
        )
        result = bootstrap_grouped_predictions(predictions, n_bootstrap=20)
        self.assertEqual(len(result), 10)
        self.assertEqual(set(result["method"]), {"method_a", "method_b"})
        self.assertEqual(set(result["seed"]), {0, 1})
        self.assertTrue((result["n_series"] == 4).all())


class MultiSeedStatisticsTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for method, values in {
            "method_a": [0.80, 0.82, 0.84],
            "method_b": [0.81, 0.835, 0.848],
        }.items():
            for seed, value in enumerate(values):
                rows.append(
                    {
                        "dataset": "synthetic",
                        "method": method,
                        "seed": seed,
                        "metric": "f1",
                        "value": value,
                    }
                )
        self.results = pd.DataFrame(rows)

    def test_mean_and_sample_standard_deviation(self):
        summary = calculate_multiseed_summary(
            self.results, metrics=("f1",)
        )
        row = summary[summary["method"] == "method_a"].iloc[0]
        self.assertAlmostEqual(row["mean"], 0.82)
        self.assertAlmostEqual(row["std"], 0.02)
        self.assertEqual(row["n_seeds"], 3)
        self.assertIn("±", row["mean_std"])

    def test_all_three_significance_tests_are_exported(self):
        result = run_significance_tests(
            self.results,
            "method_a",
            "method_b",
            metrics=("f1",),
        )
        self.assertEqual(
            set(result["test"]),
            {"paired_t_test", "welch_t_test", "wilcoxon_signed_rank"},
        )
        self.assertTrue((result["n_a"] == 3).all())
        self.assertTrue((result["n_b"] == 3).all())

    def test_identical_values_have_wilcoxon_p_one(self):
        identical = self.results.copy()
        identical.loc[identical["method"] == "method_b", "value"] = [0.80, 0.82, 0.84]
        result = run_significance_tests(
            identical,
            "method_a",
            "method_b",
            metrics=("f1",),
        )
        wilcoxon = result[result["test"] == "wilcoxon_signed_rank"].iloc[0]
        self.assertEqual(wilcoxon["p_value"], 1.0)
        self.assertIn("差值均为零", wilcoxon["note"])

    def test_missing_requested_seed_is_rejected(self):
        incomplete = self.results[self.results["seed"] != 2]
        with self.assertRaisesRegex(ValueError, "缺少随机种子"):
            calculate_multiseed_summary(incomplete, metrics=("f1",))


if __name__ == "__main__":
    unittest.main()
