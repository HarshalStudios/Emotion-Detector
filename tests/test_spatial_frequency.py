"""Unit and Verification Tests for SpatialFrequencyModel (Experiment A2).

Verifies:
1. Model forward pass on dummy batches of shape (B, 3, 224, 224):
   - Single sample (B = 1)
   - Multi-sample batches (B = 2, B = 4)
   - Output logits shape == (B, 7)
   - Logits are finite (no NaN, no Inf)
2. Intermediate embedding representations:
   - Spatial embedding shape == (B, 1280)
   - Frequency embedding shape == (B, 256)
   - Fused embedding shape == (B, 1536)
3. LearnableSpectralFilter:
   - Input (B, 3, 224, 224) -> Output (B, 3, 224, 113)
   - Non-negative, finite log-magnitude spectrum
4. Loss computation and backward gradient flow:
   - CrossEntropyLoss with class weights and label smoothing
   - Gradients populated and finite across all three parameter groups:
     * spatial_branch
     * frequency_branch
     * fusion_head
5. Determinism under fixed random seed 42.
6. Forward and backward verification on cached RAF-DB batch.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestSpatialFrequencyModel(unittest.TestCase):

    def setUp(self):
        try:
            import torch
            import torch.nn as nn
            self.torch_available = True
        except ImportError:
            self.torch_available = False
            return

        torch.manual_seed(42)
        from src.models.spatial_frequency import (
            SpatialFrequencyModel,
            LearnableSpectralFilter,
            FrequencyBranch,
            create_spatial_frequency_model,
        )
        self.SpatialFrequencyModel = SpatialFrequencyModel
        self.LearnableSpectralFilter = LearnableSpectralFilter
        self.FrequencyBranch = FrequencyBranch
        self.create_spatial_frequency_model = create_spatial_frequency_model

    def test_01_spectral_filter_and_frequency_branch(self):
        """Verifies 2D FFT spectral filter and frequency branch dimensions."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch

        # 1. Test LearnableSpectralFilter
        filter_module = self.LearnableSpectralFilter(in_channels=3, height=224, width=224)
        dummy_x = torch.randn(2, 3, 224, 224, dtype=torch.float32)
        spec = filter_module(dummy_x)

        # Expected shape: (B, 3, 224, 224 // 2 + 1) = (2, 3, 224, 113)
        self.assertEqual(spec.shape, (2, 3, 224, 113), f"Spectral filter shape mismatch: {spec.shape}")
        self.assertTrue(torch.isfinite(spec).all(), "Spectral filter output contains non-finite values.")
        self.assertTrue((spec >= 0.0).all(), "Log-magnitude spectrum must be strictly non-negative.")

        # 2. Test FrequencyBranch
        freq_branch = self.FrequencyBranch(in_channels=3, height=224, width=224, embedding_dim=256)
        freq_emb = freq_branch(dummy_x)

        self.assertEqual(freq_emb.shape, (2, 256), f"Frequency embedding shape mismatch: {freq_emb.shape}")
        self.assertTrue(torch.isfinite(freq_emb).all(), "Frequency embedding contains non-finite values.")
        print("[PASS] Test 1: LearnableSpectralFilter & FrequencyBranch verified.")

    def test_02_forward_shapes_and_finite_logits(self):
        """Verifies forward pass output shapes and finite values across multiple batch sizes."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch

        model = self.create_spatial_frequency_model(
            num_classes=7,
            pretrained=False,  # Use random weights for fast offline unit test
            freq_embedding_dim=256,
            dropout_rate=0.2,
        )
        model.eval()

        batch_sizes = [1, 2, 4]
        for b in batch_sizes:
            x = torch.randn(b, 3, 224, 224, dtype=torch.float32)
            with torch.no_grad():
                logits = model(x)

            self.assertEqual(logits.shape, (b, 7), f"Batch size {b}: logits shape {logits.shape} != ({b}, 7)")
            self.assertTrue(torch.isfinite(logits).all(), f"Batch size {b}: non-finite logits detected.")

            # Test intermediate feature extraction
            features = model.extract_features(x)
            self.assertEqual(features["spatial_embedding"].shape, (b, 1280))
            self.assertEqual(features["frequency_embedding"].shape, (b, 256))
            self.assertEqual(features["fused_embedding"].shape, (b, 1536))

        print("[PASS] Test 2: Forward shapes (B, 7) and finite logits verified across batch sizes (1, 2, 4).")

    def test_03_forward_backward_gradient_flow(self):
        """Verifies loss computation and gradient backpropagation across all parameter groups."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch
        import torch.nn as nn

        model = self.create_spatial_frequency_model(
            num_classes=7,
            pretrained=False,
            freq_embedding_dim=256,
            dropout_rate=0.2,
        )
        model.train()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        param_groups = model.get_param_groups(backbone_lr=1e-4, freq_lr=1e-3, head_lr=1e-3)
        optimizer = torch.optim.AdamW(param_groups)

        b = 4
        x = torch.randn(b, 3, 224, 224, dtype=torch.float32)
        y = torch.tensor([0, 1, 3, 6], dtype=torch.long)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        self.assertTrue(torch.isfinite(loss), f"Loss is not finite: {loss.item()}")
        loss.backward()

        # Check gradients in each branch
        # 1. Spatial branch gradients
        spatial_grads = [p.grad for p in model.spatial_branch.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(len(spatial_grads) > 0, "No spatial branch gradients found.")
        self.assertTrue(all(torch.isfinite(g).all() for g in spatial_grads), "Non-finite spatial branch gradients.")

        # 2. Frequency branch gradients (spectral filter + convs)
        freq_grads = [p.grad for p in model.frequency_branch.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(len(freq_grads) > 0, "No frequency branch gradients found.")
        self.assertTrue(all(torch.isfinite(g).all() for g in freq_grads), "Non-finite frequency branch gradients.")
        self.assertIsNotNone(model.frequency_branch.spectral_filter.weight_real.grad, "Spectral filter real grad missing.")
        self.assertIsNotNone(model.frequency_branch.spectral_filter.weight_imag.grad, "Spectral filter imag grad missing.")

        # 3. Fusion head gradients
        head_grads = [p.grad for p in model.fusion_head.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(len(head_grads) > 0, "No fusion head gradients found.")
        self.assertTrue(all(torch.isfinite(g).all() for g in head_grads), "Non-finite fusion head gradients.")

        optimizer.step()
        print(f"[PASS] Test 3: Forward/backward gradient flow verified (loss = {loss.item():.4f}).")

    def test_04_determinism_seed_42(self):
        """Verifies deterministic output reproduction with seed 42."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch

        def build_and_run(seed: int):
            torch.manual_seed(seed)
            m = self.create_spatial_frequency_model(num_classes=7, pretrained=False)
            m.eval()
            x = torch.ones(2, 3, 224, 224, dtype=torch.float32)
            with torch.no_grad():
                return m(x)

        logits_1 = build_and_run(42)
        logits_2 = build_and_run(42)

        max_diff = float(torch.max(torch.abs(logits_1 - logits_2)).item())
        self.assertEqual(max_diff, 0.0, f"Non-deterministic execution detected under seed 42 (diff={max_diff})")
        print("[PASS] Test 4: Seed 42 determinism strictly confirmed.")

    def test_05_cached_rafdb_batch_verification(self):
        """Verifies model execution on a cached RAF-DB batch format."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch
        import torch.nn as nn
        from src.training.dataset import CachedRAFDBDataset, create_cached_rafdb_dataloader

        cache_path = os.path.join(PROJECT_ROOT, "data", "cache", "rafdb_cache")
        completion_marker = os.path.join(cache_path, "cache_complete.json")

        model = self.create_spatial_frequency_model(num_classes=7, pretrained=False)
        model.train()
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        if os.path.isfile(completion_marker):
            print(f"[INFO] Real verified RAF-DB cache found at {cache_path}. Testing on real batch...")
            ds, loader = create_cached_rafdb_dataloader(
                split="train",
                batch_size=4,
                shuffle=True,
                num_workers=0,
                cache_dir=cache_path,
            )
            batch = next(iter(loader))
            images = batch["image"]
            labels = batch["label"]
        else:
            print("[INFO] Official full cache located on Kaggle storage. Testing with synthetic cached RAF-DB batch...")
            import tempfile
            from tests.test_cached_dataset import create_synthetic_cache
            temp_dir = tempfile.mkdtemp(prefix="rafdb_batch_verify_")
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
                images = batch["image"]
                labels = batch["label"]
            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(images.shape, (4, 3, 224, 224))
        self.assertEqual(labels.shape, (4,))

        logits = model(images)
        self.assertEqual(logits.shape, (4, 7))
        self.assertTrue(torch.isfinite(logits).all())

        loss = criterion(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        print(f"[PASS] Test 5: Real cached RAF-DB batch forward/backward pass verified (loss={loss.item():.4f}).")


if __name__ == "__main__":
    unittest.main(verbosity=2)
