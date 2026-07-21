import sys
from pathlib import Path

import numpy as np
import torch
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from degradation import DEGRADATION_NAMES, DegradationTransform, apply_degradation, load_degradation_config


class DegradationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_degradation_config(ROOT / "src/configs/degradation.yaml")

    def test_each_degradation_preserves_shape_range_and_tensor(self):
        patch = torch.linspace(0, 1, 3 * 64 * 64).reshape(3, 64, 64)
        for name in DEGRADATION_NAMES:
            with self.subTest(name=name):
                output = DegradationTransform(name, 3, self.config, seed=7)(patch)
                self.assertTrue(torch.is_tensor(output))
                self.assertEqual(output.shape, patch.shape)
                self.assertEqual(output.dtype, patch.dtype)
                self.assertGreaterEqual(float(output.min()), 0.0)
                self.assertLessEqual(float(output.max()), 1.0)

    def test_noise_is_reproducible_with_seed(self):
        patch = np.full((3, 64, 64), 0.5, dtype=np.float32)
        first = DegradationTransform("gaussian_noise", 2, self.config, seed=42)(patch)
        second = DegradationTransform("gaussian_noise", 2, self.config, seed=42)(patch)
        np.testing.assert_allclose(first, second)

    def test_invalid_blur_kernel_raises(self):
        patch = np.zeros((3, 8, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            apply_degradation(patch, "gaussian_blur", {"kernel_size": 4})


if __name__ == "__main__":
    unittest.main()
