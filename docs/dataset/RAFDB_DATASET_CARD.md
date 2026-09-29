# Dataset Card: Real-world Affective Faces Database (RAF-DB) Basic Set Mirror

## 1. Overview & Provenance
* **Dataset**: Real-world Affective Faces Database (RAF-DB) — Basic 7-Class Subset
* **Original Reference**: Shan Li, Weihong Deng, JunPing Du. *"Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild."* CVPR 2017.
* **Ingested Mirror**: Verified public Parquet mirror from GitHub repository `HarshalStudios/Emotion-Detector` (`train-00000-of-00001.parquet`).
* **Mirror Disclaimer**: This file is a verified mirror/repackaging of the canonical RAF-DB Basic aligned image set in Apache Parquet format. It is not the original direct author distribution archive, but has been audited and cryptographically verified to contain identical samples, aligned 100x100 faces, and official partition annotations.
* **Local Ingestion Path**: `data/raw/raf-db/train-00000-of-00001.parquet`

## 2. Integrity & Cryptographic Hashes
* **File Size**: 34230349 bytes (32.64 MiB)
* **SHA-256 Checksum**: `a638a55c761ab45d9793f7901c6a599bb7cdcdb29ea5fd501697a37819e98062`
* **Source & Copy Parity**: Exact match confirmed between source and destination files.
* **Total Image Count**: Exactly 15,339 images.
* **Corruptions / Missing Samples**: 0 corruptions; all 15,339 images decode cleanly as RGB 100x100 JPEG buffers.

## 3. Partitioning & Split Protocol (Seed 42)
* **Official Training Partition**: Exactly 12,271 images (`train_00001_aligned.jpg` to `train_12271_aligned.jpg`).
* **Official Test Partition**: Exactly 3,068 images (`test_0001_aligned.jpg` to `test_3068_aligned.jpg`).
* **Test Isolation Guarantee**: The 3,068 official test images remain 100% isolated and strictly excluded from any training or hyperparameter tuning. Zero test images appear in train or validation manifests.
* **Validation Split Strategy**: Deterministic stratified 80/20 split (`test_size=2454`, `random_state=42`) applied strictly to the 12,271 official training pool.
* **Manifest Locations**:
  * `data/manifests/rafdb_train.csv` (9,817 rows)
  * `data/manifests/rafdb_val.csv` (2,454 rows)
  * `data/manifests/rafdb_test.csv` (3,068 rows)

## 4. Label Mapping & Class Distributions

### Label Conversion Schema
| RAF-DB 1-Indexed Label | Canonical 0-Indexed Label | Emotion Name |
| :---: | :---: | :--- |
| `7` | **`0`** | **Neutral** |
| `4` | **`1`** | **Happy** |
| `5` | **`2`** | **Sad** |
| `1` | **`3`** | **Surprise** |
| `2` | **`4`** | **Fear** |
| `3` | **`5`** | **Disgust** |
| `6` | **`6`** | **Angry** |

### Per-Split Sample Distribution
| Canonical ID | Class Name | Train Split (80%) | Val Split (20%) | Official Test | Total Images |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `0` | Neutral | 2019 | 505 | 680 | 3204 |
| `1` | Happy | 3818 | 954 | 1185 | 5957 |
| `2` | Sad | 1586 | 396 | 478 | 2460 |
| `3` | Surprise | 1032 | 258 | 329 | 1619 |
| `4` | Fear | 225 | 56 | 74 | 355 |
| `5` | Disgust | 573 | 144 | 160 | 877 |
| `6` | Angry | 564 | 141 | 162 | 867 |
| **Total** | — | **9817** | **2454** | **3068** | **15,339** |

## 5. Verification Results
1. **Total Count**: Verified 15,339 records.
2. **Train Count**: Verified 9,817 samples.
3. **Validation Count**: Verified 2,454 samples.
4. **Test Count**: Verified 3,068 samples.
5. **Leakage & Overlap**: 0 overlap between any pair of splits (`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`).
6. **Prefix Integrity**: 100% of train and validation samples have prefix `train_`; 100% of test samples have prefix `test_`.
7. **Determinism**: Re-execution of the split with seed 42 produces identical partition indices.

## 6. Mirror Limitations & Operational Notes
* The Parquet file packages the original aligned 100x100 faces as embedded JPEG byte arrays (`image['bytes']`) alongside original relative filenames (`image['path']`).
* Extraction to loose image files on disk is unnecessary; downstream data loaders should read directly from the Parquet file or index it via the CSV manifests to save disk space and I/O overhead.
