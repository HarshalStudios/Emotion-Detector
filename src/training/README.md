# Training Engine & Dataset Module

This directory implements the dataset loading, preprocessing caching, and training engine for facial emotion recognition experiments (Baseline A0 and proposed branches).

---

## 1. Overview of Dataset Loaders

The codebase provides two complementary dataset implementations in `src/training/dataset.py`:

| Feature | `CachedRAFDBDataset` (Recommended for Training) | `RAFDBDataset` (Online Fallback / Audit) |
| :--- | :--- | :--- |
| **Source Data** | Preprocessed offline `.npz` cache (`data/cache/rafdb_cache/`) | Raw RAF-DB Parquet file (`data/raw/raf-db/train-00000-of-00001.parquet`) |
| **Preprocessing Overhead** | **0 ms / sample** (Precomputed landmarks, alignment, and crops) | ~25–50 ms / sample (Real-time MediaPipe face detection & landmarking) |
| **MediaPipe Dependency** | **Bypassed completely** during training | Evaluated on every sample via `FacePipeline` |
| **Throughput** | **> 1,500 samples/sec** (Disk / memory bound) | ~20–40 samples/sec (CPU bound on MediaPipe) |
| **Image Transform** | Fast on-the-fly ImageNet normalization from cached uint8 RGB | Full alignment warp, crop, resize, and ImageNet normalization |
| **Geometry Vector** | Pre-extracted 62-D float32 vector loaded directly | Computed via MediaPipe blendshapes + landmark distance ratios |
| **Use Case** | Multi-epoch CPU/GPU model training and hyperparameter search | Ground-truth verification, pipeline smoke tests, raw Parquet audit |

---

## 2. Cached Training Workflow

### Step 1: Offline Cache Generation (One-Time)
The offline cache is deterministically generated using `scripts/build_rafdb_cache.py`:
```bash
python scripts/build_rafdb_cache.py \
    --parquet data/raw/raf-db/train-00000-of-00001.parquet \
    --output data/cache/rafdb_cache
```

### Step 2: Cache Audit & Verification
Before initiating any training runs, verify cache integrity using `scripts/verify_rafdb_cache.py`:
```bash
python scripts/verify_rafdb_cache.py --cache data/cache/rafdb_cache
```
*Audit criteria:*
- `cache_complete.json` and `metadata.json` present.
- Split sample counts match: **Train = 9,817**, **Val = 2,454**, **Test = 3,068** (Total = 15,339).
- Images: `uint8`, shape `(224, 224, 3)`, RGB.
- Geometry: `float32`, shape `(62,)`.
- Labels: integer in `0..6`.
- Splits are completely disjoint with zero overlapping samples.

### Step 3: Training with Cached Dataset
In `configs/baseline.yaml`, enable cached dataset mode:
```yaml
data:
  use_cache: true
  cache_dir: "data/cache/rafdb_cache"
  parquet_path: "data/raw/raf-db/train-00000-of-00001.parquet"
  train_manifest: "data/manifests/rafdb_train.csv"
  val_manifest: "data/manifests/rafdb_val.csv"
  test_manifest: "data/manifests/rafdb_test.csv"
```

Instantiate the cached DataLoader directly in Python:
```python
from src.training.dataset import create_cached_rafdb_dataloader

train_ds, train_loader = create_cached_rafdb_dataloader(
    split="train",
    batch_size=32,
    shuffle=True,
    num_workers=0,
    cache_dir="data/cache/rafdb_cache",
)
```

---

## 3. Sample Dictionary Schema

Both `CachedRAFDBDataset` and `RAFDBDataset` yield standardized dictionaries for batch compatibility with `BaselineTrainer`:

```python
{
    "image": Tensor[float32, (3, 224, 224)],      # ImageNet normalized (CHW)
    "label": int,                                 # Canonical class index (0..6)
    "geometry": Tensor[float32, (62,)],           # 52 FACS blendshapes + 10 normalized distance ratios
    "geometry_valid": bool,                       # True if face detected; False on fallback
    "face_detected": bool,                        # Detection indicator
    "fallback_used": bool,                        # True if dataset-mode fallback crop was applied
    "image_path": str,                            # Source image path identifier
    "source_path": str,                           # Canonical source path (e.g. train_00001_aligned.jpg)
    "split": str,                                 # 'train', 'val', or 'test'
    "class_name": str,                            # Canonical emotion name ('Neutral', 'Happy', etc.)
    "sample_id": int,                             # Split-relative integer sample ID
}
```

---

## 4. Normalization Invariance

Cached samples store compact uint8 RGB images `(224, 224, 3)`. At data loading time, `CachedRAFDBDataset` converts:
$$\text{image}_{\text{normalized}} = \frac{\frac{\text{image}_{\text{uint8}}}{255.0} - \mu_{\text{ImageNet}}}{\sigma_{\text{ImageNet}}}$$
where:
- $\mu_{\text{ImageNet}} = [0.485, 0.456, 0.406]$
- $\sigma_{\text{ImageNet}} = [0.229, 0.224, 0.225]$

This guarantees mathematical equivalence ($\Delta < 10^{-6}$) between online `FacePipeline` tensor outputs and offline cached tensor outputs.
