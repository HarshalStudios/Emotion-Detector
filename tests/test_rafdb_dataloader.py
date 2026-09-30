"""Smoke test for RAF-DB Dataset and DataLoader.

Verifies:
1. Exact split lengths:
   - Train = 9817
   - Validation = 2454
   - Test = 3068
2. Per-sample inspection (at least 8 samples per split):
   - image is a torch.Tensor
   - image shape == (3, 224, 224)
   - image dtype == torch.float32
   - label is integer in range 0..6
   - no sample is silently discarded
   - fallback flag and detection states are valid booleans
3. DataLoader batch integrity:
   - batch_size = 4
   - shuffle = True for train, False for val/test
   - num_workers = 0
   - image batch shape == (4, 3, 224, 224)
   - label batch shape == (4,)
4. Fallback handling verification:
   - Evaluates fallback response contract (image tensor preserved, fallback_used flag True, geometry zeroed).
5. Source Parquet immutability:
   - Computes SHA-256 before and after test execution to guarantee zero mutation.
"""

import hashlib
import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.face_pipeline import FacePipeline
from src.training.dataset import RAFDBDataset, create_rafdb_dataloader


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


class TestRAFDBDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parquet_path = os.path.join(PROJECT_ROOT, "data/raw/raf-db/train-00000-of-00001.parquet")
        cls.expected_sha256 = "a638a55c761ab45d9793f7901c6a599bb7cdcdb29ea5fd501697a37819e98062"

        # Record initial SHA-256
        cls.initial_sha256 = compute_sha256(cls.parquet_path)
        assert cls.initial_sha256 == cls.expected_sha256, (
            f"Initial Parquet SHA256 mismatch: {cls.initial_sha256} vs {cls.expected_sha256}"
        )

        # Shared FacePipeline instance
        cls.pipeline = FacePipeline()

        # Instantiate datasets
        cls.train_dataset = RAFDBDataset(split="train", pipeline=cls.pipeline)
        cls.val_dataset = RAFDBDataset(split="val", pipeline=cls.pipeline)
        cls.test_dataset = RAFDBDataset(split="test", pipeline=cls.pipeline)

    def test_01_dataset_lengths(self):
        print("\n--- Test 1: Verifying Dataset Lengths ---")
        train_len = len(self.train_dataset)
        val_len = len(self.val_dataset)
        test_len = len(self.test_dataset)

        print(f"Train dataset length: {train_len} (expected 9817)")
        print(f"Val dataset length:   {val_len} (expected 2454)")
        print(f"Test dataset length:  {test_len} (expected 3068)")

        self.assertEqual(train_len, 9817, f"Expected 9817, got {train_len}")
        self.assertEqual(val_len, 2454, f"Expected 2454, got {val_len}")
        self.assertEqual(test_len, 3068, f"Expected 3068, got {test_len}")
        self.assertEqual(train_len + val_len + test_len, 15339)
        print("PASS: Dataset lengths strictly verified.")

    def test_02_sample_loading_and_contract(self):
        print("\n--- Test 2: Verifying 8 Samples Per Split ---")
        splits = [
            ("train", self.train_dataset),
            ("val", self.val_dataset),
            ("test", self.test_dataset),
        ]

        total_tested = 0
        fallback_counts = {"train": 0, "val": 0, "test": 0}

        for split_name, ds in splits:
            print(f"Testing split '{split_name}' (first 8 samples)...")
            for i in range(8):
                sample = ds[i]
                total_tested += 1

                # 1. Image is PyTorch Tensor
                img = sample["image"]
                self.assertIsInstance(img, torch.Tensor, f"Sample {i} in {split_name} is not a torch.Tensor")

                # 2. Shape is (3, 224, 224)
                self.assertEqual(img.shape, (3, 224, 224), f"Sample {i} shape {img.shape} != (3, 224, 224)")

                # 3. Dtype is float32
                self.assertEqual(img.dtype, torch.float32, f"Sample {i} dtype {img.dtype} != float32")

                # 4. Label is integer 0..6
                label = sample["label"]
                self.assertIsInstance(label, int, f"Sample {i} label {label} is not int")
                self.assertIn(label, range(7), f"Sample {i} label {label} not in 0..6")

                # 5. Geometry tensor
                geom = sample["geometry"]
                self.assertIsInstance(geom, torch.Tensor)
                self.assertEqual(geom.shape, (62,))
                self.assertEqual(geom.dtype, torch.float32)

                # 6. Fallback and detection flags
                self.assertIsInstance(sample["face_detected"], bool)
                self.assertIsInstance(sample["fallback_used"], bool)
                self.assertIsInstance(sample["geometry_valid"], bool)

                if sample["fallback_used"]:
                    fallback_counts[split_name] += 1
                    # In fallback: geometry should be zeroed and invalid
                    self.assertFalse(sample["geometry_valid"])
                    self.assertTrue(torch.all(geom == 0.0))

            print(f"PASS: 8 samples verified for {split_name} (fallbacks observed: {fallback_counts[split_name]}).")

        self.assertEqual(total_tested, 24)

    def test_03_dataloader_batches(self):
        print("\n--- Test 3: Verifying DataLoader Batch Shapes ---")
        splits = [
            ("train", self.train_dataset, True),
            ("val", self.val_dataset, False),
            ("test", self.test_dataset, False),
        ]

        for split_name, ds, shuffle in splits:
            loader = DataLoader(ds, batch_size=4, shuffle=shuffle, num_workers=0)
            batch = next(iter(loader))

            imgs = batch["image"]
            labels = batch["label"]

            self.assertIsInstance(imgs, torch.Tensor)
            self.assertEqual(imgs.shape, (4, 3, 224, 224), f"Batch images shape {imgs.shape} != (4, 3, 224, 224)")
            self.assertEqual(imgs.dtype, torch.float32)

            self.assertIsInstance(labels, torch.Tensor)
            self.assertEqual(labels.shape, (4,), f"Batch labels shape {labels.shape} != (4,)")

            print(f"PASS: {split_name} batch shape = {imgs.shape}, labels = {labels.shape}")

    def test_04_fallback_behavior_handling(self):
        print("\n--- Test 4: Verifying Fallback Behavior Handling ---")
        # Test synthetic no-face input directly through FacePipeline in dataset mode (is_webcam=False)
        import numpy as np
        blank_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        res = self.pipeline.process_image(blank_rgb, is_webcam=False)

        self.assertFalse(res["face_detected"])
        self.assertTrue(res["fallback_used"])
        self.assertFalse(res["partial_face"])
        self.assertIsNotNone(res["image_tensor"])
        self.assertEqual(res["image_tensor"].shape, (3, 224, 224))
        self.assertFalse(res["geometry_valid"])
        self.assertTrue(np.all(res["geometry_vector"] == 0.0))
        print("PASS: FacePipeline fallback contract confirmed.")

    def test_05_source_parquet_immutability(self):
        print("\n--- Test 5: Verifying Source Parquet Immutability ---")
        final_sha256 = compute_sha256(self.parquet_path)
        print(f"Final SHA-256:   {final_sha256}")
        print(f"Expected SHA-256: {self.expected_sha256}")
        self.assertEqual(
            final_sha256,
            self.expected_sha256,
            f"Source Parquet was mutated! Expected {self.expected_sha256}, got {final_sha256}",
        )
        print("PASS: Source Parquet byte integrity preserved 100%.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
