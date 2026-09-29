# Tests Directory

Scope & Boundary:
This directory will hold automated unit and regression tests:
- `test_parity.py`: Preprocessing parity unit tests (PyTorch vs. Inference tensor identity) and PyTorch vs. ONNX Runtime inference numerical parity (`max_abs_err < 1e-4`).
- `test_pipeline.py`: Face detection edge cases (no face, multiple faces, partial face bounding box intersections, occlusions).
- `test_branches.py`: Tensor shape and dimensionality checks across Spatial, Frequency, Geometry, and Gated Fusion layers.

*Note: In accordance with Step 1/Step 2 research scope, no code implementation occurs until research plan and pipeline thresholds are formally approved.*
