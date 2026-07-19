import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prototype_bank import cosine_kmeans


class PrototypeBankTests(unittest.TestCase):
    def test_cosine_kmeans_separates_opposite_directions(self):
        features = np.array(
            [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]],
            dtype=np.float32,
        )
        assignments, centers, iterations = cosine_kmeans(features, 2, seed=3)
        self.assertEqual(assignments[0], assignments[1])
        self.assertEqual(assignments[2], assignments[3])
        self.assertNotEqual(assignments[0], assignments[2])
        self.assertTrue(np.allclose(np.linalg.norm(centers, axis=1), 1.0))
        self.assertGreaterEqual(iterations, 1)


if __name__ == "__main__":
    unittest.main()
