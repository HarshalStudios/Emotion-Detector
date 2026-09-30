"""RAF-DB Dataset and DataLoader Module.

Implements RAFDBDataset adhering to the locked Step 1 & 2 Preprocessing Contract:
- Reads canonical manifest CSVs (train, val, test).
- Retrieves source image bytes from the RAF-DB Parquet file.
- Integrates the existing FacePipeline without duplicating preprocessing logic.
- Enforces dataset mode (is_webcam=False):
  - Webcam-only partial-face rejection is strictly bypassed.
  - If MediaPipe detects no face, uses deterministic dataset-mode fallback (bilinear resize to 224x224, ImageNet normalization, zero-masked 62-D geometry).
  - Exposes fallback_used, face_detected, and geometry_valid flags in the return dictionary.
  - Zero samples are silently discarded.
- Returns PyTorch Tensors:
  - "image": Tensor[float32, 3, 224, 224] ImageNet normalized
  - "label": int in range 0..6
  - "geometry": Tensor[float32, 62]
  - "geometry_valid": bool
  - "face_detected": bool
  - "fallback_used": bool
  - "image_path": str
  - "class_name": str
"""

import csv
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing.face_pipeline import FacePipeline

logger = logging.getLogger(__name__)

# In-memory cached lookup for Parquet image bytes: path -> bytes
_PARQUET_STORE_CACHE: Dict[str, Dict[str, bytes]] = {}


def get_parquet_image_store(parquet_path: str) -> Dict[str, bytes]:
    """
    Loads and caches all raw image bytes from the RAF-DB Parquet file.
    The Parquet file is ~34MB, making an in-memory dictionary lookup (~35MB)
    extremely fast (sub-millisecond O(1) random access per item).
    """
    abs_parquet = os.path.abspath(parquet_path)
    if abs_parquet not in _PARQUET_STORE_CACHE:
        if not os.path.exists(abs_parquet):
            raise FileNotFoundError(f"Parquet source not found at {abs_parquet}")
        table = pq.read_table(abs_parquet)
        store: Dict[str, bytes] = {}
        img_col = table["image"]
        num_rows = len(table)
        for i in range(num_rows):
            item = img_col[i].as_py()
            store[item["path"]] = item["bytes"]
        _PARQUET_STORE_CACHE[abs_parquet] = store
        logger.info(f"Loaded and indexed {len(store)} images from {abs_parquet}")
    return _PARQUET_STORE_CACHE[abs_parquet]


class RAFDBDataset(Dataset):
    """
    PyTorch Dataset for RAF-DB Facial Emotion Recognition.
    """

    def __init__(
        self,
        split: Optional[str] = None,
        manifest_path: Optional[str] = None,
        parquet_path: str = "data/raw/raf-db/train-00000-of-00001.parquet",
        pipeline: Optional[FacePipeline] = None,
        transform: Optional[Any] = None,
    ):
        """
        Args:
            split: Split name ('train', 'val', or 'test').
            manifest_path: Explicit path to manifest CSV. If None, resolves from split.
            parquet_path: Path to RAF-DB Parquet file.
            pipeline: Optional FacePipeline instance. If None, instantiates a default FacePipeline.
            transform: Optional additional PyTorch transform applied to image tensor.
        """
        super().__init__()
        if manifest_path is None:
            if split is None:
                raise ValueError("Either 'split' or 'manifest_path' must be provided.")
            manifest_path = os.path.join("data", "manifests", f"rafdb_{split}.csv")

        self.manifest_path = manifest_path
        self.parquet_path = parquet_path
        self.split = split
        self.transform = transform

        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_path}")

        # Load manifest rows
        self.samples: List[Dict[str, Any]] = []
        with open(self.manifest_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "image_path": row["image_path"],
                    "original_raf_label": int(row["original_raf_label"]),
                    "canonical_label": int(row["canonical_label"]),
                    "class_name": row["class_name"],
                    "split": row.get("split", self.split or ""),
                })

        # Load Parquet image store (shared memory cache)
        self.image_store = get_parquet_image_store(self.parquet_path)

        # Initialize or reuse FacePipeline
        if pipeline is None:
            self.pipeline = FacePipeline()
        else:
            self.pipeline = pipeline

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves sample at index idx.
        Returns:
            Dict containing:
                "image": torch.Tensor of shape (3, 224, 224), float32, ImageNet-normalized
                "label": int in range 0..6 (canonical class index)
                "geometry": torch.Tensor of shape (62,), float32
                "geometry_valid": bool
                "face_detected": bool
                "fallback_used": bool
                "image_path": str
                "class_name": str
        """
        record = self.samples[idx]
        image_path = record["image_path"]

        if image_path not in self.image_store:
            raise KeyError(f"Image {image_path} not found in Parquet store.")

        raw_bytes = self.image_store[image_path]

        # Decode JPEG bytes to BGR then convert to RGB
        buf = np.frombuffer(raw_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to decode image from bytes for {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Execute preprocessing with is_webcam=False (dataset mode)
        # This guarantees:
        # 1. Partial-face rejection is disabled.
        # 2. If MediaPipe detects face: 5-pt alignment -> 1.30 margin crop -> 224x224 resize -> ImageNet norm.
        # 3. If zero faces detected: direct 224x224 resize -> ImageNet norm, zero-geometry, fallback_used=True.
        # 4. Zero samples are silently discarded.
        result = self.pipeline.process_image(img_rgb, is_webcam=False)

        image_np = result["image_tensor"]
        if image_np is None:
            raise RuntimeError(f"Unexpected None image_tensor for {image_path} in dataset mode.")

        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image_np).float()
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        geom_np = result.get("geometry_vector", np.zeros(62, dtype=np.float32))
        geometry_tensor = torch.from_numpy(geom_np).float()

        label = record["canonical_label"]

        return {
            "image": image_tensor,
            "label": label,
            "geometry": geometry_tensor,
            "geometry_valid": bool(result.get("geometry_valid", False)),
            "face_detected": bool(result.get("face_detected", False)),
            "fallback_used": bool(result.get("fallback_used", False)),
            "image_path": image_path,
            "class_name": record["class_name"],
        }


def create_rafdb_dataloader(
    split: str,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pipeline: Optional[FacePipeline] = None,
    manifest_path: Optional[str] = None,
    parquet_path: str = "data/raw/raf-db/train-00000-of-00001.parquet",
) -> Tuple[RAFDBDataset, DataLoader]:
    """
    Factory function to instantiate RAFDBDataset and corresponding PyTorch DataLoader.
    """
    if shuffle is None:
        shuffle = (split == "train")

    dataset = RAFDBDataset(
        split=split,
        manifest_path=manifest_path,
        parquet_path=parquet_path,
        pipeline=pipeline,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )

    return dataset, loader
