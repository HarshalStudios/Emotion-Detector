# Models Module: Facial Emotion Recognition Architectures

This directory contains the neural network architecture definitions for the progressive experiment series:

---

## 1. Experiment A0: Spatial Baseline (`spatial_baseline.py`)
- **Backbone Pool**:
  - `convnext_tiny` (28.59M parameters)
  - `efficientnet_b0` (5.29M parameters)
  - `mobilenet_v3_large` (4.20M backbone parameters, 5.48M full model) — current development default
- **Input**: Image tensor $(B, 3, 224, 224)$ float32, ImageNet normalized.
- **Output**: 7 canonical emotion logits $(B, 7)$.

---

## 2. Experiment A2: Spatial + Frequency Dual-Branch (`spatial_frequency.py`)
Combines deep spatial representation with a 2D Fourier frequency domain representation to capture both high-level semantic facial cues and subtle, high-frequency textural/crease variations.

### Architecture Overview
1. **Spatial Branch (`MobileNetV3-Large`)**:
   - Pretrained ImageNet weights.
   - Final 1000-class linear classification layer replaced with `nn.Identity()`.
   - **Output**: Spatial embedding tensor $(B, 1280)$.
   - **Parameters**: **4,202,032** parameters.

2. **Frequency Branch (`FrequencyBranch`)**:
   - **Input**: Raw normalized image $(B, 3, 224, 224)$.
   - **Learnable 2D Spectral Filter (`LearnableSpectralFilter`)**:
     - Real-input 2D FFT (`torch.fft.rfft2`, `norm="ortho"`) $\to$ Complex spectrum $(B, 3, 224, 113)$.
     - Complex learnable filter weights $W = W_r + i \cdot W_i \in \mathbb{C}^{3 \times 224 \times 113}$ initialized near identity ($W_r \approx 1.0, W_i \approx 0.0$).
     - Elementwise filtering: $X_{\text{filtered}} = X \odot W$.
     - Log-magnitude dynamic range compression: $S = \log(1 + |X_{\text{filtered}}|)$ $\to (B, 3, 224, 113)$.
     - Filter parameters: **151,872** parameters.
   - **Convolutional Spectral Encoder**:
     - Stage 1: `Conv2d(3, 32, k=3, s=2, p=1)` + `BatchNorm2d` + `GELU` $\to (B, 32, 112, 57)$
     - Stage 2: `Conv2d(32, 64, k=3, s=2, p=1)` + `BatchNorm2d` + `GELU` $\to (B, 64, 56, 29)$
     - Stage 3: `Conv2d(64, 128, k=3, s=2, p=1)` + `BatchNorm2d` + `GELU` $\to (B, 128, 28, 15)$
     - Stage 4: `Conv2d(128, 256, k=3, s=2, p=1)` + `BatchNorm2d` + `GELU` $\to (B, 256, 14, 8)$
     - Global average pooling: `AdaptiveAvgPool2d((1, 1))` + `Flatten` $\to (B, 256)$
     - Linear projection: `Linear(256, 256)` + `LayerNorm(256)` + `GELU` + `Dropout(0.2)` $\to (B, 256)$
     - Encoder parameters: **455,200** parameters.
   - **Total Frequency Branch Parameters**: **607,072** parameters.
   - **Output**: Compact frequency embedding $(B, 256)$.

3. **Fusion & Classification Head (`fusion_head`)**:
   - Concatenation: $\text{Concat}(\text{spatial}, \text{frequency}) \to (B, 1536)$.
   - Multi-Layer Perceptron:
     - `Linear(1536, 512)` (786,944 parameters)
     - `LayerNorm(512)` (1,024 parameters)
     - `Hardswish()`
     - `Dropout(p=0.2)`
     - `Linear(512, 7)` (3,591 parameters)
   - **Total Fusion Head Parameters**: **791,559** parameters.
   - **Output**: Logits $(B, 7)$.

### Total Parameter Summary (Experiment A2)
| Branch / Component | Input Tensor | Output Tensor | Parameter Count |
| :--- | :--- | :--- | :--- |
| **Spatial Branch** (`MobileNetV3-Large`) | $(B, 3, 224, 224)$ | $(B, 1280)$ | 4,202,032 |
| **Frequency Branch** (Spectral + Conv) | $(B, 3, 224, 224)$ | $(B, 256)$ | 607,072 |
| **Fusion & Classification Head** | $(B, 1536)$ | $(B, 7)$ | 791,559 |
| **Total Model (A2)** | $(B, 3, 224, 224)$ | $(B, 7)$ | **5,600,663 (~5.60M)** |

---

## 3. Strict Experimental Isolation
- **Experiment A0**: Spatial-only baseline (`SpatialBaseline`).
- **Experiment A2**: Spatial + Frequency dual-branch (`SpatialFrequencyModel`).
- **Geometry**: Strictly isolated to future experiments (A3/A4/A5). No geometry inputs in A0 or A2.
