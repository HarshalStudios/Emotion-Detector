# Models Module Directory

Scope & Boundary:
This directory will contain the architectural definitions for the multi-branch model:
- `spatial_backbone.py`: Pretrained deep visual feature extractors (ConvNeXt-Tiny, MobileNetV3-Large, EfficientNet-B0)
- `frequency_branch.py`: Learnable spectral transform and 2D FFT filtering modules
- `geometry_branch.py`: MediaPipe landmark coordinate normalization and blendshape feature extractor MLP
- `fusion.py`: Gated fusion mechanism vs. naive concatenation ablation blocks
- `emotion_classifier.py`: End-to-end multi-representation model assembly
- `postprocessing.py`: Temperature scaling calibration and Exponential Moving Average (EMA) temporal smoothing

*Note: In accordance with Step 1/Step 2 research scope, no code implementation occurs until research plan and pipeline thresholds are formally approved.*
