#!/usr/bin/env python3
"""
scripts/build_rafdb_cache.py

Deterministic RAF-DB Preprocessing Cache Generator.
Processes raw RAF-DB Parquet file using the exact FacePipeline implementation
and serializes preprocessed uint8 (224, 224, 3) RGB images and 62-D geometry vectors
into a structured, versioned, and verified offline cache.

Usage:
    python scripts/build_rafdb_cache.py \\
        --parquet /path/to/train-00000-of-00001.parquet \\
        --output /path/to/cache \\
        [--limit N] \\
        [--allow-unverified-parquet]
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.face_pipeline import FacePipeline

# Authoritative Constants matching scripts/ingest_and_split_rafdb.py
EXPECTED_SHA256 = "a638a55c761ab45d9793f7901c6a599bb7cdcdb29ea5fd501697a37819e98062"
EXPECTED_BYTES = 34230349
TOTAL_IMAGES = 15339
OFFICIAL_TRAIN_POOL = 12271
OFFICIAL_TEST_COUNT = 3068
STRATIFIED_TRAIN_COUNT = 9817
STRATIFIED_VAL_COUNT = 2454
PREPROCESSING_VERSION = "rafdb_preprocess_v1"
RANDOM_SEED = 42

# Canonical Emotion Label Mapping (RAF-DB 1-indexed -> Canonical 0-indexed)
RAF_TO_CANONICAL = {
    1: (3, "Surprise"),
    2: (4, "Fear"),
    3: (5, "Disgust"),
    4: (1, "Happy"),
    5: (2, "Sad"),
    6: (6, "Angry"),
    7: (0, "Neutral"),
}

CANONICAL_TO_NAME = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Angry",
}


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file efficiently in 8MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> Optional[str]:
    """Attempts to retrieve current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
            return commit if commit else None
    except Exception:
        pass
    return None


def partition_rafdb_records(
    table: pa.Table,
    seed: int = RANDOM_SEED,
    val_size: int = STRATIFIED_VAL_COUNT,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Reads records from Parquet table and executes the exact authoritative deterministic split.
    Splits are determined on the full dataset before any optional limit is applied.
    """
    num_rows = len(table)
    train_pool: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    for i in range(num_rows):
        path = table["image"][i].as_py()["path"]
        raw_label = table["label"][i].as_py()
        raf_label = int(raw_label[0])

        if raf_label not in RAF_TO_CANONICAL:
            raise ValueError(f"Encountered unexpected RAF label {raf_label} in record {path}")

        can_label, class_name = RAF_TO_CANONICAL[raf_label]
        record = {
            "row_idx": i,
            "image_path": path,
            "original_raf_label": raf_label,
            "canonical_label": can_label,
            "class_name": class_name,
        }

        if path.startswith("train_"):
            train_pool.append(record)
        elif path.startswith("test_"):
            test_records.append(record)
        else:
            raise ValueError(f"Unrecognized file prefix in path: {path}")

    # Deterministic sorting by image_path before splitting
    train_pool.sort(key=lambda r: r["image_path"])
    test_records.sort(key=lambda r: r["image_path"])

    num_classes = len(set(r["original_raf_label"] for r in train_pool))
    if len(train_pool) >= val_size and val_size >= num_classes and num_classes > 1:
        # Exact deterministic 80/20 stratified split
        train_y = [r["original_raf_label"] for r in train_pool]
        tr_indices, val_indices = train_test_split(
            range(len(train_pool)),
            test_size=val_size,
            random_state=seed,
            stratify=train_y,
        )
        tr_indices = sorted(tr_indices)
        val_indices = sorted(val_indices)

        final_train = []
        for idx in tr_indices:
            rec = dict(train_pool[idx])
            rec["split"] = "train"
            final_train.append(rec)

        final_val = []
        for idx in val_indices:
            rec = dict(train_pool[idx])
            rec["split"] = "val"
            final_val.append(rec)
    else:
        # Fallback for synthetic/micro test tables with fewer samples than val_size
        val_count = max(1, int(len(train_pool) * 0.2)) if train_pool else 0
        final_train = [dict(r, split="train") for r in train_pool[val_count:]]
        final_val = [dict(r, split="val") for r in train_pool[:val_count]]

    final_test = []
    for r in test_records:
        rec = dict(r)
        rec["split"] = "test"
        final_test.append(rec)

    if limit is not None and limit > 0:
        print(f"[LIMIT ACTIVE] Restricting processing to first {limit} records across splits.")
        # Subsample deterministically while preserving assigned split tags
        all_ordered = final_train + final_val + final_test
        selected_subset = all_ordered[:limit]
        final_train = [r for r in selected_subset if r["split"] == "train"]
        final_val = [r for r in selected_subset if r["split"] == "val"]
        final_test = [r for r in selected_subset if r["split"] == "test"]

    return final_train, final_val, final_test


def build_cache(
    parquet_path: str,
    output_dir: str,
    limit: Optional[int] = None,
    allow_unverified_parquet: bool = False,
) -> Dict[str, Any]:
    """
    Main cache generation workflow.
    """
    start_time = datetime.datetime.now(datetime.timezone.utc)
    print("=" * 70)
    print("DETERMINISTIC RAF-DB PREPROCESSING CACHE GENERATOR")
    print(f"Started at: {start_time.isoformat()}")
    print("=" * 70)

    # 1. Validation of Parquet file
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Input Parquet file not found at: {parquet_path}")

    actual_size = os.path.getsize(parquet_path)
    print(f"Reading Parquet: {parquet_path} ({actual_size:,} bytes)")
    print("Verifying SHA-256 hash...")
    actual_sha256 = compute_sha256(parquet_path)
    print(f"Actual SHA-256:   {actual_sha256}")
    print(f"Expected SHA-256: {EXPECTED_SHA256}")

    if actual_sha256 != EXPECTED_SHA256:
        if allow_unverified_parquet or limit is not None:
            print("WARNING: Parquet SHA-256 does not match official RAF-DB hash.")
            print("Proceeding because --allow-unverified-parquet or --limit is active.")
        else:
            raise ValueError(
                f"Parquet SHA-256 mismatch!\n"
                f"Expected: {EXPECTED_SHA256}\n"
                f"Got:      {actual_sha256}\n"
                f"If you are running in test mode, pass --allow-unverified-parquet."
            )
    else:
        print("PASS: Parquet SHA-256 checksum verified perfectly.")

    # 2. Load Parquet Table
    print("\nLoading Parquet table into memory...")
    table = pq.read_table(parquet_path)
    num_rows = len(table)
    print(f"Total rows in Parquet table: {num_rows}")

    # 3. Partition into deterministic splits
    train_records, val_records, test_records = partition_rafdb_records(
        table=table,
        seed=RANDOM_SEED,
        val_size=STRATIFIED_VAL_COUNT,
        limit=limit,
    )
    total_selected = len(train_records) + len(val_records) + len(test_records)
    print(f"\nSplits to process:")
    print(f"  Train:      {len(train_records):>6}")
    print(f"  Validation: {len(val_records):>6}")
    print(f"  Test:       {len(test_records):>6}")
    print(f"  Total:      {total_selected:>6}")

    # 4. Initialize Preprocessing Pipeline
    print("\nInitializing FacePipeline (MediaPipe Vision)...")
    pipeline = FacePipeline()
    print(f"  Model path:          {pipeline.model_path}")
    print(f"  Detector model path: {pipeline.detector_model_path}")
    print(f"  Target size:         {pipeline.target_size}x{pipeline.target_size}")
    print(f"  Margin factor:       {pipeline.margin_factor}")

    # 5. Staging directory setup for atomic safety
    staging_dir = os.path.join(output_dir, f".staging_{int(start_time.timestamp())}")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    splits_data = [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]

    global_stats = {
        "face_detected": 0,
        "face_not_detected": 0,
        "fallback_used": 0,
        "geometry_valid": 0,
        "geometry_invalid": 0,
    }
    class_counts_by_split: Dict[str, Dict[str, int]] = {
        "train": {str(k): 0 for k in range(7)},
        "val": {str(k): 0 for k in range(7)},
        "test": {str(k): 0 for k in range(7)},
    }
    processed_count = 0

    print("\nProcessing records...")
    for split_name, records in splits_data:
        split_dir = os.path.join(staging_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        split_manifest_items = []

        print(f"--- Processing {split_name.upper()} split ({len(records)} samples) ---")
        for idx, rec in enumerate(records):
            row_idx = rec["row_idx"]
            image_path = rec["image_path"]
            label = rec["canonical_label"]

            # Decode JPEG image bytes directly from Parquet cell
            image_cell = table["image"][row_idx].as_py()
            img_bytes = image_cell["bytes"]

            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"Failed to decode image bytes for {image_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Process with FacePipeline in dataset mode (is_webcam=False)
            res = pipeline.process_image(img_rgb, is_webcam=False)

            face_detected = bool(res["face_detected"])
            fallback_used = bool(res["fallback_used"])
            geom_valid = bool(res["geometry_valid"])
            aligned_rgb = res["aligned_image_rgb"]
            geom_vec = res["geometry_vector"]

            # Strict contract validation
            assert aligned_rgb is not None, f"aligned_image_rgb is None for {image_path}"
            assert aligned_rgb.shape == (224, 224, 3), f"Bad shape: {aligned_rgb.shape}"
            assert aligned_rgb.dtype == np.uint8, f"Bad dtype: {aligned_rgb.dtype}"
            assert geom_vec.shape == (62,), f"Bad geom shape: {geom_vec.shape}"
            assert geom_vec.dtype == np.float32, f"Bad geom dtype: {geom_vec.dtype}"
            assert isinstance(geom_valid, bool), f"geom_valid is not bool: {type(geom_valid)}"

            # Update stats
            if face_detected:
                global_stats["face_detected"] += 1
            else:
                global_stats["face_not_detected"] += 1

            if fallback_used:
                global_stats["fallback_used"] += 1

            if geom_valid:
                global_stats["geometry_valid"] += 1
            else:
                global_stats["geometry_invalid"] += 1

            class_counts_by_split[split_name][str(label)] += 1
            processed_count += 1

            # Deterministic filename based on split and index
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            sample_filename = f"{base_name}.npz"
            sample_filepath = os.path.join(split_dir, sample_filename)

            # Serialize cached sample using np.savez_compressed
            np.savez_compressed(
                sample_filepath,
                image=aligned_rgb,
                geometry=geom_vec,
                geometry_valid=geom_valid,
                label=np.int64(label),
                source_path=image_path,
                split=split_name,
                preprocessing_version=PREPROCESSING_VERSION,
            )

            split_manifest_items.append({
                "sample_id": idx,
                "filename": sample_filename,
                "source_path": image_path,
                "label": int(label),
                "class_name": CANONICAL_TO_NAME[label],
                "geometry_valid": geom_valid,
                "fallback_used": fallback_used,
            })

            if (idx + 1) % 500 == 0 or (idx + 1) == len(records):
                print(f"  [{split_name}] {idx + 1}/{len(records)} processed "
                      f"(Detected: {global_stats['face_detected']}, Fallback: {global_stats['fallback_used']})")

        # Save split index.json
        index_path = os.path.join(split_dir, "index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(split_manifest_items, f, indent=2)

    end_time = datetime.datetime.now(datetime.timezone.utc)
    duration_sec = (end_time - start_time).total_seconds()

    # 6. Build Metadata
    metadata: Dict[str, Any] = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_parquet": {
            "path": os.path.abspath(parquet_path),
            "size_bytes": actual_size,
            "sha256": actual_sha256,
            "expected_sha256": EXPECTED_SHA256,
            "hash_verified": (actual_sha256 == EXPECTED_SHA256),
        },
        "is_limited": (limit is not None and limit > 0),
        "limit": limit,
        "counts": {
            "total_samples": processed_count,
            "train_count": len(train_records),
            "val_count": len(val_records),
            "test_count": len(test_records),
        },
        "class_counts": class_counts_by_split,
        "sample_schema": {
            "image": {
                "shape": [224, 224, 3],
                "dtype": "uint8",
                "color_space": "RGB",
                "normalized": False,
            },
            "geometry": {
                "dimension": 62,
                "dtype": "float32",
                "description": "52 FACS blendshapes + 10 normalized distance ratios",
            },
            "geometry_valid": {
                "dtype": "bool",
                "description": "True if face detected and geometry valid; False on zero-face fallback",
            },
            "label": {
                "dtype": "int64",
                "range": [0, 6],
            },
            "split": {
                "dtype": "string",
                "values": ["train", "val", "test"],
            },
            "preprocessing_version": PREPROCESSING_VERSION,
        },
        "label_mapping": {str(k): v for k, v in CANONICAL_TO_NAME.items()},
        "preprocessing_pipeline": {
            "alignment_method": "5-point roll-based rigid rotation alignment",
            "scale": 1.0,
            "margin_factor": 1.30,
            "target_size": [224, 224],
            "dataset_fallback_rule": "Direct bilinear 224x224 resize; zero geometry; geometry_valid=False; 0 samples discarded",
        },
        "geometry_statistics": {
            "geometry_valid_count": global_stats["geometry_valid"],
            "geometry_invalid_count": global_stats["geometry_invalid"],
            "face_detected_count": global_stats["face_detected"],
            "face_not_detected_count": global_stats["face_not_detected"],
            "fallback_used_count": global_stats["fallback_used"],
        },
        "reproducibility": {
            "random_seed": RANDOM_SEED,
            "git_commit": get_git_commit(),
            "created_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": round(duration_sec, 2),
        },
    }

    metadata_path = os.path.join(staging_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 7. Create explicit completion marker
    completion_marker: Dict[str, Any] = {
        "status": "COMPLETED",
        "preprocessing_version": PREPROCESSING_VERSION,
        "completed_at": end_time.isoformat(),
        "total_samples": processed_count,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "is_limited": (limit is not None and limit > 0),
        "limit": limit,
        "metadata_sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    completion_marker_path = os.path.join(staging_dir, "cache_complete.json")
    with open(completion_marker_path, "w", encoding="utf-8") as f:
        json.dump(completion_marker, f, indent=2)

    # 8. Safe atomic finalization
    os.makedirs(output_dir, exist_ok=True)
    for item in os.listdir(staging_dir):
        src_item = os.path.join(staging_dir, item)
        dst_item = os.path.join(output_dir, item)
        if os.path.exists(dst_item):
            if os.path.isdir(dst_item):
                shutil.rmtree(dst_item)
            else:
                os.remove(dst_item)
        shutil.move(src_item, dst_item)

    shutil.rmtree(staging_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("CACHE GENERATION COMPLETED SUCCESSFULLY")
    print(f"Output directory:    {os.path.abspath(output_dir)}")
    print(f"Total samples:       {processed_count}")
    print(f"Face detected:       {global_stats['face_detected']} ({global_stats['face_detected']/max(1, processed_count)*100:.1f}%)")
    print(f"Fallback used:       {global_stats['fallback_used']} ({global_stats['fallback_used']/max(1, processed_count)*100:.1f}%)")
    print(f"Geometry valid:      {global_stats['geometry_valid']}")
    print(f"Total time elapsed:  {duration_sec:.1f}s")
    print("=" * 70)

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic RAF-DB Preprocessing Cache Generator")
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/raw/raf-db/train-00000-of-00001.parquet",
        help="Path to input train-00000-of-00001.parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cache/rafdb_cache",
        help="Target output directory for the cache",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for small test/development mode (e.g. --limit 20)",
    )
    parser.add_argument(
        "--allow-unverified-parquet",
        action="store_true",
        help="Allow proceeding even if Parquet SHA-256 does not match official hash (for testing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_cache(
        parquet_path=args.parquet,
        output_dir=args.output,
        limit=args.limit,
        allow_unverified_parquet=args.allow_unverified_parquet,
    )
