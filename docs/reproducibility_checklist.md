# Reproducibility Checklist

This checklist enforces experimental rigor and auditability across all research steps in "Multi-Representation Facial Mood Analysis from a Single RGB Camera".

## 1. Environment & Hardware Lock
- [ ] Hardware specification documented (exact CPU model, GPU model, total system RAM, GPU VRAM, OS version, kernel).
- [ ] Python runtime version pinned (e.g., Python 3.10.x).
- [ ] Dependency lockfile generated (`requirements.txt` or `poetry.lock` with exact pinned versions of PyTorch, torchvision, timm, albumentations, mediapipe, onnxruntime, numpy, scikit-learn).
- [ ] Random seeds explicitly set and logged across:
  - Python `random.seed(seed)`
  - NumPy `np.random.seed(seed)`
  - PyTorch `torch.manual_seed(seed)`
  - PyTorch CUDA `torch.cuda.manual_seed_all(seed)`
  - Deterministic cuDNN: `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`
  - DataLoader worker seeding via `worker_init_fn` using `torch.initial_seed()`.

## 2. Dataset & Split Integrity
- [ ] Dataset integrity verified via SHA256 checksums of the downloaded archive and extracted file manifests.
- [ ] Official test set (3,068 images for RAF-DB) isolated and NEVER evaluated during development or model selection.
- [ ] Fixed stratified validation split: Generated from the official training split (12,271 images) using a fixed 20% stratified split (seed 42), producing 9,817 train / 2,454 validation images. Saved permanently in `data/processed/manifests/raf_db_split_s42.csv`.
- [ ] Test set isolation: The official test set and the webcam final test subset (P07–P10) are stored in protected, read-only locations and evaluated exactly ONCE at project completion after architecture freeze.
- [ ] Class mapping strictly standardized across all datasets to the 7 canonical classes:
  - 0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Angry.
  - Any ignored/non-target classes (e.g., contempt, uncertain) explicitly filtered and logged.
- [ ] Preprocessing parity verified: Identical cropping, alignment, resizing, and normalization applied during training and inference.

## 3. Training & Validation Execution
- [ ] Configuration management: Every experiment driven by an immutable YAML config file in `configs/`.
- [ ] Config committed or logged alongside checkpoint with Git commit hash and timestamp.
- [ ] Model checkpoints save full state dict, optimizer state, lr scheduler state, epoch, seed, and validation metrics.
- [ ] Early stopping decisions strictly monitored on validation macro-F1 (never training loss or test metrics).
- [ ] Never discard training samples during data loading or preprocessing; all valid dataset samples must be retained.
- [ ] Metric logging covers:
  - Macro-averaged F1, macro-averaged Precision, macro-averaged Recall
  - Per-class F1 for all 7 classes
  - Overall Accuracy and Balanced Accuracy
  - Expected Calibration Error (ECE, 15 bins)
  - Confusion Matrix saved as an artifact.

## 4. Multi-Seed Statistical Validation
- [ ] Development ablations run across 3 fixed seeds (Seed 42, 43, 44).
- [ ] Final Baseline vs. Proposed comparison run across 5 fixed seeds (42, 43, 44, 45, 46).
- [ ] Report results as `mean ± sample standard deviation` across seeds.
- [ ] Primary statistical test: Paired percentile bootstrap 95% confidence interval of the macro-F1 difference (1,000 resamples). For each bootstrap sample of validation cases, calculate the macro-F1 difference separately for each development seed, then average those 3 seed differences. Seeds are never treated as independent statistical observations. Student's t-test with 3 seeds is strictly prohibited.
- [ ] An architectural branch is retained only if:
  - Mean difference across seeds exceeds pooled seed standard deviation (mean_delta >= s_pooled), AND
  - The 95% bootstrap confidence interval strictly excludes zero (CI_lower > 0.0), AND
  - No single class F1 drops by more than 2.0 percentage points (-0.020).
- [ ] Webcam evaluation treats the human participant as the true unit of statistical independence (N=6 validation, N=4 test), reporting participant-level aggregated scores.

## 5. Deployment & Parity Verification
- [ ] ONNX export verified with dynamic or static batch dimensions as intended.
- [ ] Numerical parity test passed: PyTorch output tensor vs. ONNX Runtime output tensor with `max_abs_err < 1e-4` on 100 benchmark faces.
- [ ] CPU inference benchmark conducted strictly under standardized conditions: batch size = 1, single thread and multi-thread reported, 100 warm-up runs, minimum 500 timed runs, latency distribution reported (P50, P90, P95, P99).
- [ ] Stability metric logged: Expression flicker rate (state transitions per second under steady input) tested with and without EMA smoothing.
