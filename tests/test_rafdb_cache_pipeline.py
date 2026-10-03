"""
Unit tests for RAF-DB cache generator and verifier:
- Deterministic splitting verification
- Cache generation with synthetic data in test mode (--limit)
- Full verification pass using verify_cache()
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.build_rafdb_cache import (
    PREPROCESSING_VERSION,
    RAF_TO_CANONICAL,
    build_cache,
    partition_rafdb_records,
)
from scripts.verify_rafdb_cache import verify_cache


def create_synthetic_jpeg_bytes(width: int = 120, height: int = 120, draw_face: bool = True) -> bytes:
    """Creates in-memory JPEG bytes of either a synthetic face or blank box."""
    canvas = np.full((height, width, 3), 220, dtype=np.uint8)
    if draw_face:
        cv2.ellipse(canvas, (width // 2, height // 2), (width // 3, height // 3), 0, 0, 360, (200, 170, 150), -1)
        cv2.circle(canvas, (width // 3, height // 3), 5, (50, 40, 30), -1)
        cv2.circle(canvas, (2 * width // 3, height // 3), 5, (50, 40, 30), -1)
        cv2.line(canvas, (width // 2, height // 2), (width // 2, height // 2 + 10), (140, 100, 80), 2)
        cv2.line(canvas, (width // 3, 2 * height // 3), (2 * width // 3, 2 * height // 3), (120, 50, 50), 2)
    success, enc = cv2.imencode(".jpg", canvas)
    assert success, "Failed to encode JPEG"
    return enc.tobytes()


def generate_synthetic_rafdb_parquet(filepath: str, num_train: int = 25, num_test: int = 10) -> None:
    """Generates synthetic RAF-DB Parquet file conforming to raw RAF-DB Arrow schema."""
    image_structs = []
    labels = []

    # 1. Train pool
    for i in range(num_train):
        path = f"train_{i:05d}_aligned.jpg"
        raf_label = (i % 7) + 1  # 1..7
        jpeg_bytes = create_synthetic_jpeg_bytes(100, 100, draw_face=(i % 2 == 0))
        image_structs.append({"bytes": jpeg_bytes, "path": path})
        labels.append([raf_label])

    # 2. Test pool
    for i in range(num_test):
        path = f"test_{i:05d}_aligned.jpg"
        raf_label = (i % 7) + 1
        jpeg_bytes = create_synthetic_jpeg_bytes(100, 100, draw_face=(i % 2 == 0))
        image_structs.append({"bytes": jpeg_bytes, "path": path})
        labels.append([raf_label])

    # Construct PyArrow Table
    image_type = pa.struct([
        ("bytes", pa.binary()),
        ("path", pa.string()),
    ])
    label_type = pa.list_(pa.int64())

    table = pa.Table.from_arrays(
        [
            pa.array(image_structs, type=image_type),
            pa.array(labels, type=label_type),
        ],
        names=["image", "label"],
    )

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    pq.write_table(table, filepath)


class TestRAFDBCachePipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="rafdb_cache_test_")
        self.synthetic_parquet = os.path.join(self.temp_dir, "synthetic_rafdb.parquet")
        self.cache_output_dir = os.path.join(self.temp_dir, "cache_out")
        generate_synthetic_rafdb_parquet(self.synthetic_parquet, num_train=30, num_test=10)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_deterministic_splitting(self):
        """Verifies that split assignment is 100% deterministic and repeatable."""
        table = pq.read_table(self.synthetic_parquet)
        tr1, val1, te1 = partition_rafdb_records(table, seed=42, val_size=7)
        tr2, val2, te2 = partition_rafdb_records(table, seed=42, val_size=7)

        self.assertEqual([r["image_path"] for r in tr1], [r["image_path"] for r in tr2])
        self.assertEqual([r["image_path"] for r in val1], [r["image_path"] for r in val2])
        self.assertEqual([r["image_path"] for r in te1], [r["image_path"] for r in te2])

        # Verify disjointness
        tr_paths = set(r["image_path"] for r in tr1)
        val_paths = set(r["image_path"] for r in val1)
        te_paths = set(r["image_path"] for r in te1)
        self.assertEqual(len(tr_paths & val_paths), 0)
        self.assertEqual(len(tr_paths & te_paths), 0)
        self.assertEqual(len(val_paths & te_paths), 0)

    def test_02_build_cache_and_verify(self):
        """Builds cache with limit and runs full verify_cache() suite."""
        limit = 12
        meta = build_cache(
            parquet_path=self.synthetic_parquet,
            output_dir=self.cache_output_dir,
            limit=limit,
            allow_unverified_parquet=True,
        )

        self.assertTrue(os.path.isfile(os.path.join(self.cache_output_dir, "cache_complete.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.cache_output_dir, "metadata.json")))

        # Run verify_cache script function
        verification_passed = verify_cache(self.cache_output_dir, quick=False)
        self.assertTrue(verification_passed, "verify_cache() reported failure on freshly generated cache!")

        # Sample inspection
        with open(os.path.join(self.cache_output_dir, "metadata.json")) as f:
            metadata = json.load(f)
        self.assertEqual(metadata["preprocessing_version"], PREPROCESSING_VERSION)
        self.assertEqual(metadata["counts"]["total_samples"], limit)

        # Inspect one .npz file directly
        train_dir = os.path.join(self.cache_output_dir, "train")
        train_files = [f for f in os.listdir(train_dir) if f.endswith(".npz")]
        if train_files:
            sample_data = np.load(os.path.join(train_dir, train_files[0]))
            self.assertEqual(sample_data["image"].shape, (224, 224, 3))
            self.assertEqual(sample_data["image"].dtype, np.uint8)
            self.assertEqual(sample_data["geometry"].shape, (62,))
            self.assertEqual(sample_data["geometry"].dtype, np.float32)
            self.assertIn(int(sample_data["label"]), range(7))
            self.assertEqual(str(sample_data["preprocessing_version"]), PREPROCESSING_VERSION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
