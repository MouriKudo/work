import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pbip_train import prototype_class_logits, prototype_contrastive_loss


class PrototypeObjectiveTests(unittest.TestCase):
    def setUp(self):
        # Prototypes 0-1 are negative and 2-3 are positive.
        self.prototype_labels = torch.tensor([0, 0, 1, 1])
        self.similarities = torch.tensor(
            [
                [0.90, 0.80, 0.10, 0.20],
                [0.10, 0.20, 0.90, 0.80],
            ],
            dtype=torch.float32,
        )

    def test_class_logits_keep_negative_and_positive_evidence_separate(self):
        logits = prototype_class_logits(
            self.similarities,
            self.prototype_labels,
            top_k=2,
            temperature=1.0,
        )
        self.assertGreater(logits[0, 0].item(), logits[0, 1].item())
        self.assertGreater(logits[1, 1].item(), logits[1, 0].item())

    def test_contrastive_loss_uses_labels_and_is_non_negative(self):
        logits = prototype_class_logits(
            self.similarities,
            self.prototype_labels,
            top_k=2,
            temperature=0.2,
        )
        correct_loss = prototype_contrastive_loss(logits, torch.tensor([0, 1]))
        wrong_loss = prototype_contrastive_loss(logits, torch.tensor([1, 0]))
        self.assertGreaterEqual(correct_loss.item(), 0.0)
        self.assertGreater(wrong_loss.item(), correct_loss.item())


if __name__ == "__main__":
    unittest.main()
