import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gradcam import GradCAM


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.features(x))
        return self.fc(self.pool(x).flatten(1)).squeeze(1)


class Tiny3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv3d(1, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.features(x))
        return self.fc(self.pool(x).flatten(1)).squeeze(1)


class GradCAMTest(unittest.TestCase):
    def test_cam_shape_and_range(self):
        model = TinyModel()
        with GradCAM(model, model.features) as cam_builder:
            cams, probabilities = cam_builder(torch.rand(2, 3, 16, 16))
        self.assertEqual(tuple(cams.shape), (2, 16, 16))
        self.assertEqual(tuple(probabilities.shape), (2,))
        self.assertGreaterEqual(float(cams.min()), 0.0)
        self.assertLessEqual(float(cams.max()), 1.0)

    def test_3d_cam_shape(self):
        model = Tiny3DModel()
        with GradCAM(model, model.features) as cam_builder:
            cams, probabilities = cam_builder(torch.rand(2, 1, 5, 16, 16))
        self.assertEqual(tuple(cams.shape), (2, 5, 16, 16))
        self.assertEqual(tuple(probabilities.shape), (2,))


if __name__ == "__main__":
    unittest.main()
