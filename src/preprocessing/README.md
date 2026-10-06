# Preprocessing Module Directory

Scope & Boundary:
This directory will contain the single, unified preprocessing pipeline shared identically across:
1. Training data loaders
2. Offline evaluation scripts
3. ONNX export pipeline
4. Real-time inference API and demo

Pipeline stages:
- Face detection & canonical landmark localization
- 5-point roll-based rigid rotation alignment
- Margin expansion and square aspect-ratio cropping
- Bilinear resizing to 224x224
- Partial-face validation & filtering
- Dual output generation: RGB normalized tensor + Frequency branch tensor + Geometry feature vector

*Note: In accordance with Step 1/Step 2 research scope, no code implementation occurs until research plan and pipeline thresholds are formally approved.*
