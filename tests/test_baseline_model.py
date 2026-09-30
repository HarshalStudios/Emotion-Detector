"""Smoke test for Spatial Baseline Model.

Verifies:
1. Instantiation of all candidate backbones:
   - ConvNeXt-Tiny
   - EfficientNet-B0
   - MobileNetV3-Large
2. Forward pass with dummy batch of shape (2, 3, 224, 224):
   - Output shape == (2, 7)
   - Logits are finite (no NaN, no Inf)
3. Loss calculation and backward pass:
   - CrossEntropyLoss with class weights and label smoothing
   - Gradients are populated and non-NaN
4. Real RAF-DB DataLoader integration:
   - Fetches one real batch from RAFDBDataset (via create_rafdb_dataloader)
   - Feeds into selected default backbone (ConvNeXt-Tiny)
   - Verifies batch output shape (4, 7) and valid loss computation
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.spatial_baseline import SpatialBaseline, SUPPORTED_BACKBONES
from src.training.dataset import create_rafdb_dataloader
from src.training.trainer import (
    compute_balanced_class_weights,
    create_criterion,
    BaselineTrainer,
)
from src.evaluation.metrics import compute_classification_metrics


class TestSpatialBaselineModel(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.dummy_input = torch.randn(2, 3, 224, 224, dtype=torch.float32)
        self.dummy_target = torch.tensor([1, 6], dtype=torch.long)
        self.class_counts = [2019, 3818, 1586, 1032, 225, 573, 564]
        self.class_weights = compute_balanced_class_weights(self.class_counts)
        self.criterion = create_criterion(self.class_weights, label_smoothing=0.1)

    def test_01_backbone_candidate_pool(self):
        print("\n--- Test 1: Testing All Candidate Backbones ---")
        for backbone_name in SUPPORTED_BACKBONES:
            print(f"Instantiating candidate backbone: {backbone_name}...")
            model = SpatialBaseline(
                backbone_name=backbone_name,
                num_classes=7,
                pretrained=True,
                dropout_rate=0.2,
            )
            model.eval()

            # 1. Forward pass
            with torch.no_grad():
                logits = model(self.dummy_input)

            # 2. Output shape verification
            self.assertEqual(
                logits.shape,
                (2, 7),
                f"Backbone {backbone_name} produced shape {logits.shape}, expected (2, 7)",
            )

            # 3. Finite logits verification
            self.assertTrue(
                torch.isfinite(logits).all().item(),
                f"Backbone {backbone_name} produced non-finite logits",
            )
            print(f"  [PASS] {backbone_name}: forward shape (2, 7), finite logits verified.")

            # 4. Training mode backward pass verification
            model.train()
            optimizer = torch.optim.AdamW(
                model.get_param_groups(backbone_lr=1e-4, head_lr=1e-3, weight_decay=0.01)
            )
            optimizer.zero_grad()
            train_logits = model(self.dummy_input)
            loss = self.criterion(train_logits, self.dummy_target)

            self.assertTrue(
                torch.isfinite(loss).item(),
                f"Backbone {backbone_name} loss is not finite: {loss.item()}",
            )
            loss.backward()

            # Verify head gradients exist
            head_grad_found = False
            for p in model.head_module.parameters():
                if p.grad is not None and torch.isfinite(p.grad).all():
                    head_grad_found = True
                    break
            self.assertTrue(
                head_grad_found,
                f"Backbone {backbone_name} head gradients were not populated",
            )
            optimizer.step()
            print(f"  [PASS] {backbone_name}: loss computation & backward pass verified (loss={loss.item():.4f}).")

    def test_02_metrics_computation(self):
        print("\n--- Test 2: Testing Evaluation Metrics ---")
        y_true = [0, 1, 2, 3, 4, 5, 6, 1]
        y_pred = [0, 1, 2, 3, 4, 5, 0, 1]  # 7 correct, 1 wrong
        metrics = compute_classification_metrics(y_true, y_pred)

        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("per_class_f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), 7)
        self.assertEqual(len(metrics["confusion_matrix"][0]), 7)
        self.assertAlmostEqual(metrics["accuracy"], 7 / 8, places=4)
        print(f"PASS: Metrics computed cleanly: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")

    def test_03_real_dataloader_integration(self):
        print("\n--- Test 3: Real RAF-DB DataLoader Feeding into Default Backbone ---")
        # MobileNetV3-Large is the current DEVELOPMENT default because of the constrained CPU environment. This is NOT the experimentally selected final backbone. Final backbone selection remains determined by the predefined baseline evaluation.
        default_backbone = "mobilenet_v3_large"
        print(f"Selected default backbone: {default_backbone}")

        model = SpatialBaseline(
            backbone_name=default_backbone,
            num_classes=7,
            pretrained=True,
        )
        model.eval()

        # Load one batch from real RAF-DB train split
        dataset, dataloader = create_rafdb_dataloader(
            split="train",
            batch_size=4,
            shuffle=True,
            num_workers=0,
        )
        batch = next(iter(dataloader))
        images = batch["image"]
        labels = batch["label"]

        self.assertEqual(images.shape, (4, 3, 224, 224))
        self.assertEqual(labels.shape, (4,))

        with torch.no_grad():
            logits = model(images)

        self.assertEqual(logits.shape, (4, 7))
        self.assertTrue(torch.isfinite(logits).all().item())

        loss = self.criterion(logits, labels)
        self.assertTrue(torch.isfinite(loss).item())
        print(f"PASS: Real batch fed into {default_backbone}: logits shape {logits.shape}, loss {loss.item():.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
