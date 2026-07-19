import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_experiments import METHODS, parse_float_list, parse_int_list, parse_method_list


class ExperimentRunnerTests(unittest.TestCase):
    def test_default_seed_matrix(self):
        self.assertEqual(parse_int_list("0,1,2"), [0, 1, 2])

    def test_all_methods_expands_to_four_methods(self):
        self.assertEqual(parse_method_list("all"), list(METHODS))
        self.assertEqual(len(METHODS), 4)

    def test_beta_sweep_parser(self):
        self.assertEqual(parse_float_list("0.05,0.1"), [0.05, 0.1])


if __name__ == "__main__":
    unittest.main()
