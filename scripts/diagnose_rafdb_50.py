"""
Diagnostic Script: 50 RAF-DB Images Preprocessing Diagnostic.
Verifies that detected + fallback = 50 and discarded = 0.
"""

import os
import io
import csv
import pyarrow.parquet as pq
from PIL import Image
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing.face_pipeline import FacePipeline

def run_diagnostic():
    print("=" * 65)
    print("50 RAF-DB TRAINING IMAGES PREPROCESSING DIAGNOSTIC")
    print("=" * 65)

    pipeline = FacePipeline()

    parquet_path = "data/raw/raf-db/train-00000-of-00001.parquet"
    train_manifest = "data/manifests/rafdb_train.csv"

    assert os.path.exists(parquet_path), f"Parquet missing: {parquet_path}"
    assert os.path.exists(train_manifest), f"Manifest missing: {train_manifest}"

    # Read exactly the first 50 image paths from rafdb_train.csv
    target_paths = []
    with open(train_manifest, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target_paths.append(row["image_path"])
            if len(target_paths) == 50:
                break

    print(f"Loaded {len(target_paths)} training image paths from manifest.")
    print(f"First 3 paths: {target_paths[:3]}")
    print(f"Last 3 paths:  {target_paths[-3:]}\n")

    # Read parquet into table
    table = pq.read_table(parquet_path)
    
    # Create quick path-to-index lookup for parquet rows
    path_to_row = {}
    for i in range(len(table)):
        p = table["image"][i].as_py()["path"]
        path_to_row[p] = i

    detected_count = 0
    fallback_count = 0
    tensors_count = 0
    discarded_count = 0
    geom_valid_count = 0

    per_sample_log = []

    for idx, path in enumerate(target_paths):
        row_idx = path_to_row[path]
        img_bytes = table["image"][row_idx].as_py()["bytes"]
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_arr = np.array(pil_img)

        res = pipeline.process_image(img_arr, is_webcam=False)

        is_detected = res["face_detected"]
        is_fallback = res["fallback_used"]
        has_tensor = res["image_tensor"] is not None
        is_geom_valid = res["geometry_valid"]

        if is_detected:
            detected_count += 1
        if is_fallback:
            fallback_count += 1
        if has_tensor:
            tensors_count += 1
            assert res["image_tensor"].shape == (3, 224, 224)
            assert res["image_tensor"].dtype == np.float32
        else:
            discarded_count += 1

        if is_geom_valid:
            geom_valid_count += 1

        per_sample_log.append({
            "idx": idx,
            "path": path,
            "face_detected": is_detected,
            "fallback_used": is_fallback,
            "has_tensor": has_tensor,
            "geometry_valid": is_geom_valid,
        })

    print("-" * 65)
    print("DIAGNOSTIC SUMMARY (N = 50):")
    print(f"  Detected by MediaPipe:            {detected_count}")
    print(f"  Dataset fallback used:            {fallback_count}")
    print(f"  Total successfully converted:     {tensors_count}")
    print(f"  Discarded samples:                {discarded_count}")
    print(f"  Geometry valid count:             {geom_valid_count}")
    print("-" * 65)

    assert detected_count + fallback_count == 50, f"FAILED: detected ({detected_count}) + fallback ({fallback_count}) != 50"
    assert tensors_count == 50, f"FAILED: tensors_count ({tensors_count}) != 50"
    assert discarded_count == 0, f"FAILED: discarded_count ({discarded_count}) != 0"
    print("VERIFICATION CHECK: detected + fallback == 50 [PASSED]")
    print("VERIFICATION CHECK: discarded == 0           [PASSED]")
    print("=" * 65)

if __name__ == "__main__":
    run_diagnostic()
