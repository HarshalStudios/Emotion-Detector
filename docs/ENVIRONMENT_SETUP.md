# Environment Setup & Reproduction Guide

This guide provides the complete, deterministic procedure to reproduce the project development and runtime environment on a persistent CPU-only Linux system.

---

## 1. Supported Python Runtime
- **Primary Supported Versions**: Python **3.10.x** or **3.11.x** (verified in production container with Python 3.10.12 / 3.11).
- **Environment Isolation**: It is strongly recommended to use a virtual environment:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

---

## 2. System-Level Dependencies (OS Packages)
MediaPipe's vision tasks and landmarkers require dynamic EGL/GLES shared libraries even when operating in CPU mode under headless Linux.

On Debian/Ubuntu systems, install the required runtime libraries:
```bash
sudo apt-get update && sudo apt-get install -y \
    libegl1 \
    libgles2
```
*Note: Using `opencv-python-headless` in Python avoids the need for GUI libraries like `libgl1-mesa-glx` or `libx11`.*

---

## 3. CPU-Only PyTorch Installation
**CRITICAL**: Standard PyPI `pip install torch` downloads approximately 2.5 GB of CUDA, cuDNN, and Triton binaries that are unneeded on CPU systems and can cause memory or disk exhaustion.

Install CPU-only builds of PyTorch and torchvision directly from the official PyTorch CPU wheel repository:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 4. General Python Dependencies
Install the remaining project dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Dependency Breakdown:
- **`numpy>=1.24.0,<2.0.0`**: Core multi-dimensional array operations.
- **`scikit-learn>=1.3.0`**: Classification metrics (macro-F1, weighted-F1, confusion matrix, balanced class weights).
- **`opencv-python-headless>=4.8.0`**: Image decoding, geometric transformations, and bounding box crops without GUI window dependencies.
- **`mediapipe>=0.10.9,<0.10.15`**: Face detection and 478-point 3D face mesh landmarking pipeline with fallback handling.
- **`Pillow>=9.5.0`**: Image I/O utilities.
- **`pyarrow>=14.0.0`**: Fast parquet file reading for the packaged RAF-DB dataset (`data/raw/raf-db/train-00000-of-00001.parquet`).
- **`PyYAML>=6.0`**: Parsing model and baseline training experiment configurations (`configs/baseline.yaml`).
- **`onnxruntime>=1.16.0`**: CPU inference latency benchmarking and verification.
- **`psutil>=5.9.0`**: System resource inspection used by benchmark scripts.

---

## 5. Compatibility & Version Constraints Discovered During Development
1. **NumPy 2.x Incompatibility with MediaPipe**:
   - MediaPipe `0.10.x` C-extensions were compiled against the NumPy 1.x C ABI.
   - Installing NumPy `2.0+` results in `RuntimeError: NumPy 2.x ABI is incompatible` or segmentation faults upon calling `mediapipe.tasks.python.vision`.
   - **Constraint**: `numpy>=1.24.0,<2.0.0` is strictly enforced in `requirements.txt`.
2. **Headless OpenCV**:
   - `opencv-python-headless` must be used instead of `opencv-python` to prevent errors relating to missing X11/Wayland display servers on containerized and server environments.
3. **CPU-Only Wheels**:
   - Installing `torch` without `--index-url https://download.pytorch.org/whl/cpu` pulls large GPU packages which may lead to out-of-disk failures in memory-constrained environments.
4. **PyArrow Versioning**:
   - `pyarrow` is required to deserialize RAF-DB dataset parquet columns directly into memory.

---

## 6. Verification Commands

To verify each installed package individually, execute:

### 1. NumPy
```bash
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
```

### 2. OpenCV
```bash
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
```

### 3. MediaPipe
```bash
python3 -c "import mediapipe as mp; print('MediaPipe:', mp.__version__)"
```

### 4. PyTorch (Verify CPU mode)
```bash
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

### 5. torchvision
```bash
python3 -c "import torchvision; print('torchvision:', torchvision.__version__)"
```

### 6. pyarrow
```bash
python3 -c "import pyarrow; print('pyarrow:', pyarrow.__version__)"
```

### 7. scikit-learn
```bash
python3 -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

### All-in-One Verification Script
```bash
python3 -c "
import numpy as np
import cv2
import mediapipe as mp
import torch
import torchvision
import pyarrow
import sklearn

print('--- All Dependencies Verified Successfully ---')
print(f'NumPy:        {np.__version__}')
print(f'OpenCV:       {cv2.__version__}')
print(f'MediaPipe:    {mp.__version__}')
print(f'PyTorch:      {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'torchvision:  {torchvision.__version__}')
print(f'PyArrow:      {pyarrow.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
"
```

---

## 7. Final Project Smoke Tests

Once all dependencies are installed, run the unit and integration smoke tests:

1. **Verify Baseline Model & Backbones**:
   ```bash
   python3 tests/test_baseline_model.py
   ```
   *Verifies candidate backbone instantiation (ConvNeXt-Tiny, EfficientNet-B0, MobileNetV3-Large), dummy forward/loss/backward passes, metric calculations, and real RAF-DB DataLoader feeding.*

2. **Verify RAF-DB DataLoader & Parquet Streaming**:
   ```bash
   python3 tests/test_rafdb_dataloader.py
   ```
   *Verifies zero-copy parquet access, stratification, 7-class label integrity, and deterministic batch iteration.*

3. **Verify Preprocessing & Face Crop Pipeline**:
   ```bash
   python3 tests/test_face_pipeline_fallback.py
   ```
   *Verifies MediaPipe face detection, landmarking, and heuristic aspect-ratio bounding box fallback logic.*
