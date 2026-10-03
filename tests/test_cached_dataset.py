"""Smoke and Verification Tests for CachedRAFDBDataset.

Verifies:
1. Sample loading:
   - image shape == (3, 224, 224)
   - image dtype == float32
   - geometry shape == (62,)
   - geometry dtype == float32
   - label is integer in range 0..6
   - geometry_valid is boolean
   - metadata keys present and correct
2. Mathematical correctness of on-the-fly ImageNet normalization.
3. Split loading and split isolation across train, val, and test.
4. Cached mode does NOT invoke MediaPipe or FacePipeline.
5. PyTorch DataLoader batching (when torch is available).
6. Full-cache split count verification (when full cache exists on disk: 9817 / 2454 / 3068).
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.dataset import (
    CachedRAFDBDataset,
    CANONICAL_CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    create_cached_rafdb_dataloader,
)


def create_synthetic_cache(root_dir: str, num_per_split: int = 10) -> None:
    """Helper to create a valid synthetic cached dataset structure for unit tests."""
    splits = ["train", "val", "test"]

    for split in splits:
        split_dir = os.path.join(root_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        index_items = []

        for i in range(num_per_split):
            base_name = f"{split}_{i:05d}_aligned"
            filename = f"{base_name}.npz"
            file_path = os.path.join(split_dir, filename)

            # Known synthetic image with distinct pixel values
            # Channel 0 (R)=100, Channel 1 (G)=150, Channel 2 (B)=200
            img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
            img_rgb[:, :, 0] = 100
            img_rgb[:, :, 1] = 150
            img_rgb[:, :, 2] = 200

            # 62-D geometry vector
            geom_vec = np.linspace(0.0, 1.0, 62, dtype=np.float32)
            geom_valid = (i % 2 == 0)
            label = i % 7

            np.savez_compressed(
                file_path,
                image=img_rgb,
                geometry=geom_vec,
                geometry_valid=geom_valid,
                label=np.int64(label),
                source_path=f"{base_name}.jpg",
                split=split,
                preprocessing_version="rafdb_preprocess_v1",
            )

            index_items.append({
                "sample_id": i,
                "filename": filename,
                "source_path": f"{base_name}.jpg",
                "label": int(label),
                "class_name": CANONICAL_CLASS_NAMES[label],
                "geometry_valid": geom_valid,
                "fallback_used": not geom_valid,
            })

        with open(os.path.join(split_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index_items, f, indent=2)


class TestCachedRAFDBDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_cached_dataset_")
        cls.cache_root = os.path.join(cls.temp_dir, "rafdb_cache")
        create_synthetic_cache(cls.cache_root, num_per_split=14)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_sample_loading_and_shapes(self):
        """Verifies sample loading, shapes, dtypes, and fields across splits."""
        for split in ["train", "val", "test"]:
            dataset = CachedRAFDBDataset(split=split, cache_dir=self.cache_root)
            self.assertEqual(len(dataset), 14, f"Split {split} count mismatch")

            sample = dataset[0]

            # 1. Image checks
            img = sample["image"]
            # Convert to numpy if tensor for uniform verification
            img_np = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
            self.assertEqual(img_np.shape, (3, 224, 224), f"Image shape is {img_np.shape}, expected (3, 224, 224)")
            self.assertEqual(img_np.dtype, np.float32, f"Image dtype is {img_np.dtype}, expected float32")

            # 2. Geometry checks
            geom = sample["geometry"]
            geom_np = geom.numpy() if hasattr(geom, "numpy") else np.asarray(geom)
            self.assertEqual(geom_np.shape, (62,), f"Geometry shape is {geom_np.shape}, expected (62,)")
            self.assertEqual(geom_np.dtype, np.float32, f"Geometry dtype is {geom_np.dtype}, expected float32")

            # 3. Label and validity checks
            label = sample["label"]
            self.assertIsInstance(label, int)
            self.assertIn(label, range(7), f"Label {label} not in canonical range 0..6")
            self.assertIsInstance(sample["geometry_valid"], bool)
            self.assertIsInstance(sample["fallback_used"], bool)
            self.assertIsInstance(sample["face_detected"], bool)

            # 4. Metadata checks
            self.assertEqual(sample["split"], split)
            self.assertTrue(sample["source_path"].endswith(".jpg"))
            self.assertEqual(sample["class_name"], CANONICAL_CLASS_NAMES[label])
            self.assertEqual(sample["sample_id"], 0)

        print("[PASS] Test 1: Sample loading, shapes, and contract verified.")

    def test_02_imagenet_normalization_mathematical_accuracy(self):
        """Verifies that ImageNet normalization matches the exact formula."""
        dataset = CachedRAFDBDataset(split="train", cache_dir=self.cache_root)
        sample = dataset[0]
        img = sample["image"]
        img_np = img.numpy() if hasattr(img, "numpy") else np.asarray(img)

        # In setUpClass, uint8 RGB was solid: R=100, G=150, B=200
        expected_r = (100.0 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        expected_g = (150.0 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        expected_b = (200.0 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

        np.testing.assert_allclose(img_np[0, :, :], expected_r, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(img_np[1, :, :], expected_g, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(img_np[2, :, :], expected_b, rtol=1e-5, atol=1e-5)

        print("[PASS] Test 2: ImageNet normalization mathematical accuracy verified.")

    def test_03_split_loading_and_isolation(self):
        """Verifies that splits are isolated and invalid split names raise ValueError."""
        # Valid splits
        ds_tr = CachedRAFDBDataset(split="train", cache_dir=self.cache_root)
        ds_val = CachedRAFDBDataset(split="val", cache_dir=self.cache_root)
        ds_te = CachedRAFDBDataset(split="test", cache_dir=self.cache_root)

        self.assertEqual(ds_tr.split, "train")
        self.assertEqual(ds_val.split, "val")
        self.assertEqual(ds_te.split, "test")

        # Invalid split raises ValueError
        with self.assertRaises(ValueError):
            CachedRAFDBDataset(split="unknown_split", cache_dir=self.cache_root)

        # Missing cache dir raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            CachedRAFDBDataset(split="train", cache_dir="/nonexistent/cache/dir")

        print("[PASS] Test 3: Split loading and isolation verified.")

    def test_04_no_mediapipe_invoked_in_cached_mode(self):
        """Guarantees that CachedRAFDBDataset does NOT instantiate or invoke MediaPipe/FacePipeline."""
        # Patch FacePipeline to raise an exception if it is ever instantiated
        with patch("src.preprocessing.face_pipeline.FacePipeline", side_effect=AssertionError("FacePipeline invoked!")):
            # Instantiating and accessing CachedRAFDBDataset must NOT invoke FacePipeline
            dataset = CachedRAFDBDataset(split="train", cache_dir=self.cache_root)
            sample = dataset[0]
            self.assertIsNotNone(sample["image"])
            self.assertIsNotNone(sample["geometry"])

        print("[PASS] Test 4: Zero MediaPipe invocation in cached mode verified.")

    def test_05_dataloader_batching(self):
        """Verifies DataLoader batching if PyTorch is installed."""
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:
            print("[INFO] Test 5: PyTorch not installed in this environment; skipping DataLoader test as allowed.")
            return

        dataset, loader = create_cached_rafdb_dataloader(
            split="train",
            batch_size=4,
            shuffle=False,
            num_workers=0,
            cache_dir=self.cache_root,
        )

        self.assertIsNotNone(loader, "DataLoader should not be None when PyTorch is installed")
        batch = next(iter(loader))

        images = batch["image"]
        labels = batch["label"]
        geometry = batch["geometry"]

        self.assertIsInstance(images, torch.Tensor)
        self.assertEqual(images.shape, (4, 3, 224, 224))
        self.assertEqual(images.dtype, torch.float32)

        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(labels.shape, (4,))

        self.assertIsInstance(geometry, torch.Tensor)
        self.assertEqual(geometry.shape, (4, 62))

        print("[PASS] Test 5: DataLoader batching verified with PyTorch.")

    def test_06_full_dataset_counts_when_cache_exists(self):
        """Verifies full dataset counts (9817/2454/3068) if the official cache exists on disk."""
        official_cache_dir = os.path.join(PROJECT_ROOT, "data", "cache", "rafdb_cache")
        completion_marker = os.path.join(official_cache_dir, "cache_complete.json")

        if os.path.isfile(completion_marker):
            print(f"[INFO] Official RAF-DB cache detected at {official_cache_dir}. Running full audit...")
            ds_train = CachedRAFDBDataset(split="train", cache_dir=official_cache_dir)
            ds_val = CachedRAFDBDataset(split="val", cache_dir=official_cache_dir)
            ds_test = CachedRAFDBDataset(split="test", cache_dir=official_cache_dir)

            self.assertEqual(len(ds_train), 9817, f"Train count {len(ds_train)} != 9817")
            self.assertEqual(len(ds_val), 2454, f"Val count {len(ds_val)} != 2454")
            self.assertEqual(len(ds_test), 3068, f"Test count {len(ds_test)} != 3068")
            self.assertEqual(len(ds_train) + len(ds_val) + len(ds_test), 15339)
            print("[PASS] Test 6: Official full cache counts (9817/2454/3068) strictly verified.")
        else:
            print("[INFO] Test 6: Official full cache not present on local disk (generated on Kaggle). Check skipped.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
