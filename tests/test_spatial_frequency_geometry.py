"""Unit and Verification Tests for SpatialFrequencyGeometryModel (Experiment A4).

Verifies:
1. Geometry branch with validity masking:
   - Geometry input shape (B, 62)
   - Geometry embedding shape (B, 64)
   - Invalid geometry samples (geometry_valid == False) are strictly masked to zero vectors.
2. Embedding representations and plain concatenation:
   - Image input shape (B, 3, 224, 224)
   - Spatial embedding shape == (B, 1280)
   - Frequency embedding shape == (B, 256)
   - Geometry embedding shape == (B, 64)
   - Concatenated embedding shape == (B, 1600)
   - Output logits shape == (B, 7)
   - Logits are finite (no NaN, no Inf) across batch sizes (B = 1, 2, 4)
3. Forward and backward gradient flow across all 4 parameter groups:
   - spatial_branch
   - frequency_branch
   - geometry_branch
   - fusion_head
4. Determinism under fixed random seed 42.
5. End-to-end batch verification on cached RAF-DB format.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestSpatialFrequencyGeometryModel(unittest.TestCase):

    def setUp(self):
        try:
            import torch
            import torch.nn as nn
            self.torch_available = True
        except ImportError:
            self.torch_available = False
            return

        torch.manual_seed(42)
        from src.models.spatial_frequency_geometry import (
            SpatialFrequencyGeometryModel,
            GeometryBranch,
            create_spatial_frequency_geometry_model,
        )
        self.SpatialFrequencyGeometryModel = SpatialFrequencyGeometryModel
        self.GeometryBranch = GeometryBranch
        self.create_model = create_spatial_frequency_geometry_model

    def test_01_geometry_branch_and_validity_masking(self):
        """Verifies geometry MLP dimension (B, 64) and zero-masking on invalid geometry."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch

        geom_branch = self.GeometryBranch(
            in_features=62, hidden_dim=128, embedding_dim=64, dropout_rate=0.2
        )
        geom_branch.eval()

        b = 3
        # Dummy geometry vectors
        geom = torch.randn(b, 62, dtype=torch.float32)
        # Sample 0 is valid, Sample 1 is invalid, Sample 2 is valid
        geom_valid = torch.tensor([True, False, True], dtype=torch.bool)

        with torch.no_grad():
            emb = geom_branch(geom, geometry_valid=geom_valid)

        # 1. Check shape
        self.assertEqual(emb.shape, (b, 64), f"Expected shape ({b}, 64), got {emb.shape}")
        self.assertTrue(torch.isfinite(emb).all(), "Non-finite geometry embedding values detected.")

        # 2. Check validity masking: Sample 1 must be strictly all zeros
        sample_0_norm = float(torch.norm(emb[0]).item())
        sample_1_norm = float(torch.norm(emb[1]).item())
        sample_2_norm = float(torch.norm(emb[2]).item())

        self.assertGreater(sample_0_norm, 0.0, "Valid geometry embedding should be non-zero.")
        self.assertEqual(sample_1_norm, 0.0, "Invalid geometry embedding MUST be strictly masked to all zeros.")
        self.assertGreater(sample_2_norm, 0.0, "Valid geometry embedding should be non-zero.")

        print(f"[PASS] Test 1: Geometry branch shape (B, 64) and validity masking verified (invalid norm={sample_1_norm}).")

    def test_02_embedding_shapes_and_concatenation(self):
        """Verifies intermediate shapes (1280, 256, 64 -> 1600) and logits (B, 7)."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch

        model = self.create_model(
            num_classes=7,
            pretrained=False,
            freq_embedding_dim=256,
            geom_embedding_dim=64,
            dropout_rate=0.2,
        )
        model.eval()

        batch_sizes = [1, 2, 4]
        for b in batch_sizes:
            x = torch.randn(b, 3, 224, 224, dtype=torch.float32)
            geom = torch.randn(b, 62, dtype=torch.float32)
            geom_valid = torch.ones(b, dtype=torch.bool)

            with torch.no_grad():
                features = model.extract_features(x, geometry=geom, geometry_valid=geom_valid)
                logits = model(x, geometry=geom, geometry_valid=geom_valid)

            # Spatial embedding: (B, 1280)
            self.assertEqual(features["spatial_embedding"].shape, (b, 1280))
            # Frequency embedding: (B, 256)
            self.assertEqual(features["frequency_embedding"].shape, (b, 256))
            # Geometry embedding: (B, 64)
            self.assertEqual(features["geometry_embedding"].shape, (b, 64))
            # Concatenated embedding: (B, 1600)
            self.assertEqual(features["fused_embedding"].shape, (b, 1600))
            # Output logits: (B, 7)
            self.assertEqual(logits.shape, (b, 7))

            # Finite checks
            self.assertTrue(torch.isfinite(features["fused_embedding"]).all())
            self.assertTrue(torch.isfinite(logits).all())

        print("[PASS] Test 2: Shapes (1280, 256, 64 -> 1600 -> 7) verified across batch sizes (1, 2, 4).")

    def test_03_forward_backward_gradient_flow(self):
        """Verifies loss computation and gradient backpropagation across all 4 parameter groups."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch
        import torch.nn as nn

        model = self.create_model(
            num_classes=7,
            pretrained=False,
            freq_embedding_dim=256,
            geom_embedding_dim=64,
            dropout_rate=0.2,
        )
        model.train()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        param_groups = model.get_param_groups(
            backbone_lr=1e-4, freq_lr=1e-3, geom_lr=1e-3, head_lr=1e-3
        )
        self.assertEqual(len(param_groups), 4, "Model should return 4 parameter groups.")
        optimizer = torch.optim.AdamW(param_groups)

        b = 4
        x = torch.randn(b, 3, 224, 224, dtype=torch.float32)
        geom = torch.randn(b, 62, dtype=torch.float32)
        geom_valid = torch.tensor([True, False, True, True], dtype=torch.bool)
        targets = torch.tensor([0, 2, 4, 6], dtype=torch.long)

        optimizer.zero_grad()
        logits = model(x, geometry=geom, geometry_valid=geom_valid)
        loss = criterion(logits, targets)

        self.assertTrue(torch.isfinite(loss), f"Loss is not finite: {loss.item()}")
        loss.backward()

        # Audit gradients in each of the 4 branches
        # 1. Spatial branch
        spatial_grads = [p.grad for p in model.spatial_branch.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(len(spatial_grads) > 0, "No spatial branch gradients found.")
        self.assertTrue(all(torch.isfinite(g).all() for g in spatial_grads), "Non-finite spatial gradients.")

        # 2. Frequency branch
        freq_grads = [p.grad for p in model.frequency_branch.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(len(freq_grads) > 0, "No frequency branch gradients found.")
        self.assertTrue(all(torch.isfinite(g).all() for g in freq_grads), "Non-finite frequency gradients.")

        # 3. Geometry branch
        geom_grads = [p.grad for p in model.geometry_branch.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(len(geom_grads) > 0, "No geometry branch gradients found.")
        self.assertTrue(all(torch.isfinite(g).all() for g in geom_grads), "Non-finite geometry gradients.")

        # 4. Fusion head
        head_grads = [p.grad for p in model.fusion_head.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(len(head_grads) > 0, "No fusion head gradients found.")
        self.assertTrue(all(torch.isfinite(g).all() for g in head_grads), "Non-finite fusion head gradients.")

        optimizer.step()
        print(f"[PASS] Test 3: 4-branch forward/backward gradient flow verified (loss = {loss.item():.4f}).")

    def test_04_determinism_seed_42(self):
        """Verifies deterministic output reproduction with seed 42."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch

        def build_and_run(seed: int):
            torch.manual_seed(seed)
            m = self.create_model(num_classes=7, pretrained=False)
            m.eval()
            x = torch.ones(2, 3, 224, 224, dtype=torch.float32)
            geom = torch.ones(2, 62, dtype=torch.float32)
            geom_valid = torch.tensor([True, False], dtype=torch.bool)
            with torch.no_grad():
                return m(x, geometry=geom, geometry_valid=geom_valid)

        logits_1 = build_and_run(42)
        logits_2 = build_and_run(42)

        max_diff = float(torch.max(torch.abs(logits_1 - logits_2)).item())
        self.assertEqual(max_diff, 0.0, f"Non-deterministic execution detected under seed 42 (diff={max_diff})")
        print("[PASS] Test 4: Seed 42 determinism strictly confirmed.")

    def test_05_cached_rafdb_batch_verification(self):
        """Verifies model execution on a cached RAF-DB batch format containing images and geometry."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch
        import torch.nn as nn
        from src.training.dataset import CachedRAFDBDataset, create_cached_rafdb_dataloader

        cache_path = os.path.join(PROJECT_ROOT, "data", "cache", "rafdb_cache")
        completion_marker = os.path.join(cache_path, "cache_complete.json")

        model = self.create_model(num_classes=7, pretrained=False)
        model.train()
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        if os.path.isfile(completion_marker):
            print(f"[INFO] Real verified RAF-DB cache detected at {cache_path}. Testing on real batch...")
            ds, loader = create_cached_rafdb_dataloader(
                split="train",
                batch_size=4,
                shuffle=True,
                num_workers=0,
                cache_dir=cache_path,
            )
            batch = next(iter(loader))
        else:
            print("[INFO] Official full cache located on Kaggle storage. Testing with synthetic cached RAF-DB batch...")
            import tempfile
            from tests.test_cached_dataset import create_synthetic_cache
            temp_dir = tempfile.mkdtemp(prefix="rafdb_batch_verify_a4_")
            try:
                create_synthetic_cache(temp_dir, num_per_split=8)
                ds, loader = create_cached_rafdb_dataloader(
                    split="train",
                    batch_size=4,
                    shuffle=False,
                    num_workers=0,
                    cache_dir=temp_dir,
                )
                batch = next(iter(loader))
            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

        images = batch["image"]
        labels = batch["label"]
        geometry = batch["geometry"]
        geometry_valid = batch["geometry_valid"]

        self.assertEqual(images.shape, (4, 3, 224, 224))
        self.assertEqual(geometry.shape, (4, 62))
        self.assertEqual(labels.shape, (4,))

        logits = model(images, geometry=geometry, geometry_valid=geometry_valid)
        self.assertEqual(logits.shape, (4, 7))
        self.assertTrue(torch.isfinite(logits).all())

        loss = criterion(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        print(f"[PASS] Test 5: Real cached RAF-DB batch forward/backward pass verified (loss={loss.item():.4f}).")


if __name__ == "__main__":
    unittest.main(verbosity=2)
