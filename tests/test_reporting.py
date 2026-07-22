import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from export_tables import (
    build_ablation_table,
    build_explainability_table,
    build_external_test_table,
    build_main_results_table,
    build_robustness_table,
    export_all_tables,
    parse_args as parse_table_args,
)
from make_figures import (
    configure_academic_style,
    draw_k_ablation,
    draw_main_metrics,
    draw_retrieval_grid,
    draw_robustness_grid,
    parse_formats,
)


class TableExportTests(unittest.TestCase):
    def test_all_five_tables_are_built_from_real_sources(self):
        main = build_main_results_table(
            PROJECT_ROOT / "runs/summary_v3/main_results.csv"
        )
        ablation = build_ablation_table(
            PROJECT_ROOT / "runs/ablations/ablation_results.csv"
        )
        robustness = build_robustness_table(
            PROJECT_ROOT / "runs/robustness/robustness_detailed.csv"
        )
        explainability = build_explainability_table(
            PROJECT_ROOT / "runs/gradcam/gradcam_samples.csv"
        )
        external = build_external_test_table(
            PROJECT_ROOT / "runs/external_test/metrics_with_ci.csv"
        )
        self.assertEqual(len(main), 4)
        self.assertEqual(len(ablation), 9)
        self.assertEqual(len(robustness), 124)
        self.assertEqual(len(explainability), 8)
        self.assertEqual(len(external), 30)
        self.assertAlmostEqual(
            main.loc[main["method"] == "pbip_lite", "f1_mean"].iloc[0],
            0.935686023201551,
        )

    def test_cli_export_writes_analysis_and_five_csv_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_table_args(
                [
                    "--output-dir",
                    str(root / "tables"),
                    "--analysis-output",
                    str(root / "results_analysis.md"),
                ]
            )
            outputs = export_all_tables(args)
            self.assertTrue(all(path.exists() for path in outputs.values()))
            self.assertEqual(len(list((root / "tables").glob("*.csv"))), 5)
            analysis = (root / "results_analysis.md").read_text(encoding="utf-8")
            self.assertIn("当前 LIDC-IDRI 数据不足", analysis)


class FigureGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_academic_style()

    def test_formats_are_strictly_validated(self):
        self.assertEqual(parse_formats("png,svg"), ("png", "svg"))
        with self.assertRaises(ValueError):
            parse_formats("png,pdf")

    def test_data_driven_figures_can_render(self):
        figures = [
            draw_main_metrics(
                PROJECT_ROOT / "runs/summary_v3/main_results_summary.csv"
            ),
            draw_k_ablation(
                PROJECT_ROOT / "runs/ablations/k_ablation_results.csv"
            ),
            draw_robustness_grid(
                PROJECT_ROOT / "runs/robustness/robustness_detailed.csv", "auc"
            ),
            draw_retrieval_grid(
                PROJECT_ROOT / "runs/retrieval/retrieval_example.csv",
                PROJECT_ROOT / "data/processed/patches",
            ),
        ]
        try:
            self.assertTrue(all(figure.axes for figure in figures))
        finally:
            for figure in figures:
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
