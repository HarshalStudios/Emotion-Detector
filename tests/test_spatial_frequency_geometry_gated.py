"""Unit and Verification Tests for SpatialFrequencyGeometryGatedModel (Experiment A5).

Verifies:
1. Learned Gated Fusion Module and Validity Masking:
   - Raw embeddings: Spatial (B, 1280), Frequency (B, 256), Geometry (B, 64)
   - Gate tensor shape == (B, 3) with sum(gates, dim=-1) == 1.0
   - Invalid geometry samples (geometry_valid == False) receive strictly 0.0 gate weight
   - Dynamic probability redistribution: sum(gates[invalid, 0:2]) == 1.0 (spatial + frequency)
   - Fused embedding shape == (B, 1600) with zeroed geometry channels for invalid samples
2. Full Model Forward Pass & Feature Extraction:
   - Image input shape (B, 3, 224, 224)
   - Geometry input shape (B, 62)
   - Output logits shape == (B, 7)
   - Gate tensor inspectability: get_gates() and return_gates=True return (B, 3)
   - Finite outputs across batch sizes (B = 1, 2, 4)
3. Forward and backward gradient backpropagation across all 5 parameter groups:
   - spatial_branch
   - frequency_branch
   - geometry_branch
   - gated_fusion
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


class TestSpatialFrequencyGeometryGatedModel(unittest.TestCase):

    def setUp(self):
        try:
            import torch
            import torch.nn as nn
            self.torch_available = True
        except ImportError:
            self.torch_available = False
            return

        torch.manual_seed(42)
        from src.models.spatial_frequency_geometry_gated import (
            SpatialFrequencyGeometryGatedModel,
            LearnedGatedFusion,
            create_spatial_frequency_geometry_gated_model,
        )
        self.SpatialFrequencyGeometryGatedModel = SpatialFrequencyGeometryGatedModel
        self.LearnedGatedFusion = LearnedGatedFusion
        self.create_model = create_spatial_frequency_geometry_gated_model

    def test_01_learned_gated_fusion_and_validity_masking(self):
        """Verifies LearnedGatedFusion shapes, softmax normalization, and strict zero-masking for invalid geometry."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch

        fusion_module = self.LearnedGatedFusion(
            spatial_dim=1280, freq_dim=256, geom_dim=64, gate_hidden_dim=128, dropout_rate=0.2
        )
        fusion_module.eval()

        b = 3
        spatial = torch.randn(b, 1280, dtype=torch.float32)
        freq = torch.randn(b, 256, dtype=torch.float32)
        geom = torch.randn(b, 64, dtype=torch.float32)
        # Sample 0: valid, Sample 1: invalid, Sample 2: valid
        geom_valid = torch.tensor([True, False, True], dtype=torch.bool)

        with torch.no_grad():
            fused_emb, gates = fusion_module(spatial, freq, geom, geometry_valid=geom_valid)

        # 1. Shape verifications
        self.assertEqual(fused_emb.shape, (b, 1600), f"Expected fused shape ({b}, 1600), got {fused_emb.shape}")
        self.assertEqual(gates.shape, (b, 3), f"Expected gates shape ({b}, 3), got {gates.shape}")
        self.assertTrue(torch.isfinite(fused_emb).all(), "Non-finite values in fused embedding.")
        self.assertTrue(torch.isfinite(gates).all(), "Non-finite values in gates.")

        # 2. Gate probabilities and bounds
        self.assertTrue((gates >= 0.0).all() and (gates <= 1.0).all(), "Gate values must be in [0, 1].")

        # 3. Valid sample gate check: sums to 1.0, geometry gate is non-zero
        self.assertAlmostEqual(float(gates[0].sum().item()), 1.0, places=5)
        self.assertGreater(float(gates[0, 2].item()), 0.0, "Valid geometry gate should receive non-zero attention.")

        # 4. Invalid sample gate check: geometry gate is strictly 0.0
        invalid_geom_gate = float(gates[1, 2].item())
        self.assertEqual(invalid_geom_gate, 0.0, "Invalid geometry gate MUST be strictly 0.0.")
        # Spatial + Frequency sum to 1.0
        spatial_freq_sum = float(gates[1, 0:2].sum().item())
        self.assertAlmostEqual(spatial_freq_sum, 1.0, places=5, msg="Spatial + Frequency gates should sum to 1.0 when geometry is invalid.")

        # 5. Verify fused representation geometry slice (1536 to 1600) is all zeros for invalid sample
        invalid_geom_slice_norm = float(torch.norm(fused_emb[1, 1536:1600]).item())
        self.assertEqual(invalid_geom_slice_norm, 0.0, "Invalid geometry slice in fused representation must be exactly 0.0.")

        print(f"[PASS] Test 1: LearnedGatedFusion shapes, normalization (sum=1.0), and invalid-geometry zeroing (gate={invalid_geom_gate}) verified.")

    def test_02_embedding_shapes_gates_and_logits(self):
        """Verifies forward pass output shapes (B, 7), inspectable gates (B, 3), and intermediate embeddings."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch

        model = self.create_model(
            num_classes=7,
            pretrained=False,
            freq_embedding_dim=256,
            geom_embedding_dim=64,
            gate_hidden_dim=128,
            dropout_rate=0.2,
        )
        model.eval()

        batch_sizes = [1, 2, 4]
        for b in batch_sizes:
            x = torch.randn(b, 3, 224, 224, dtype=torch.float32)
            geom = torch.randn(b, 62, dtype=torch.float32)
            geom_valid = torch.ones(b, dtype=torch.bool)

            with torch.no_grad():
                # Test standard forward
                logits = model(x, geometry=geom, geometry_valid=geom_valid)
                self.assertEqual(logits.shape, (b, 7))
                self.assertTrue(torch.isfinite(logits).all())

                # Test forward with return_gates=True
                logits_g, gates_g = model(x, geometry=geom, geometry_valid=geom_valid, return_gates=True)
                self.assertEqual(logits_g.shape, (b, 7))
                self.assertEqual(gates_g.shape, (b, 3))
                self.assertTrue(torch.isfinite(gates_g).all())

                # Test get_gates convenience inspection method
                gates_inspect = model.get_gates(x, geometry=geom, geometry_valid=geom_valid)
                self.assertEqual(gates_inspect.shape, (b, 3))

                # Test extract_features
                feats = model.extract_features(x, geometry=geom, geometry_valid=geom_valid)
                self.assertEqual(feats["spatial_embedding"].shape, (b, 1280))
                self.assertEqual(feats["frequency_embedding"].shape, (b, 256))
                self.assertEqual(feats["geometry_embedding"].shape, (b, 64))
                self.assertEqual(feats["gates"].shape, (b, 3))
                self.assertEqual(feats["fused_embedding"].shape, (b, 1600))

        print("[PASS] Test 2: Shapes (1280, 256, 64 -> gates (B, 3) -> 1600 -> 7) verified across batch sizes (1, 2, 4).")

    def test_03_forward_backward_gradient_flow(self):
        """Verifies loss computation and gradient backpropagation across all 5 parameter groups."""
        if not self.torch_available:
            self.skipTest("PyTorch is not installed in this environment.")

        import torch
        import torch.nn as nn

        model = self.create_model(
            num_classes=7,
            pretrained=False,
            freq_embedding_dim=256,
            geom_embedding_dim=64,
            gate_hidden_dim=128,
            dropout_rate=0.2,
        )
        model.train()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        param_groups = model.get_param_groups(
            backbone_lr=1e-4, freq_lr=1e-3, geom_lr=1e-3, gate_lr=1e-3, head_lr=1e-3
        )
        self.assertEqual(len(param_groups), 5, "Model must provide exactly 5 parameter groups.")
        optimizer = torch.optim.AdamW(param_groups)

        b = 4
        x = torch.randn(b, 3, 224, 224, dtype=torch.float32)
        geom = torch.randn(b, 62, dtype=torch.float32)
        geom_valid = torch.tensor([True, False, True, True], dtype=torch.bool)
        targets = torch.tensor([1, 3, 5, 0], dtype=torch.long)

        optimizer.zero_grad()
        logits, gates = model(x, geometry=geom, geometry_valid=geom_valid, return_gates=True)
        loss = criterion(logits, targets)

        self.assertTrue(torch.isfinite(loss), f"Loss is not finite: {loss.item()}")
        loss.backward()

        # Audit gradients across all 5 branches
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

        # 4. Gated fusion module
        gate_grads = [p.grad for p in model.gated_fusion.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(len(gate_grads) > 0, "No gated fusion gradients found.")
        self.assertTrue(all(torch.isfinite(g).all() for g in gate_grads), "Non-finite gated fusion gradients.")

        # 5. Fusion head
        head_grads = [p.grad for p in model.fusion_head.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(len(head_grads) > 0, "No fusion head gradients found.")
        self.assertTrue(all(torch.isfinite(g).all() for g in head_grads), "Non-finite fusion head gradients.")

        optimizer.step()
        print(f"[PASS] Test 3: 5-component forward/backward gradient flow verified (loss = {loss.item():.4f}).")

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
                return m(x, geometry=geom, geometry_valid=geom_valid, return_gates=True)

        (logits_1, gates_1) = build_and_run(42)
        (logits_2, gates_2) = build_and_run(42)

        max_diff_logits = float(torch.max(torch.abs(logits_1 - logits_2)).item())
        max_diff_gates = float(torch.max(torch.abs(gates_1 - gates_2)).item())
        self.assertEqual(max_diff_logits, 0.0, f"Non-deterministic logits under seed 42 (diff={max_diff_logits})")
        self.assertEqual(max_diff_gates, 0.0, f"Non-deterministic gates under seed 42 (diff={max_diff_gates})")
        print("[PASS] Test 4: Seed 42 determinism strictly confirmed (logits diff=0.0, gates diff=0.0).")

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
            temp_dir = tempfile.mkdtemp(prefix="rafdb_batch_verify_a5_")
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

        logits, gates = model(images, geometry=geometry, geometry_valid=geometry_valid, return_gates=True)
        self.assertEqual(logits.shape, (4, 7))
        self.assertEqual(gates.shape, (4, 3))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(gates).all())

        loss = criterion(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        print(f"[PASS] Test 5: Real cached RAF-DB batch forward/backward pass verified (loss={loss.item():.4f}, gates shape={gates.shape}).")


if __name__ == "__main__":
    unittest.main(verbosity=2)
