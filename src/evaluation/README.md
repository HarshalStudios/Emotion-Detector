# Evaluation Module Directory

Scope & Boundary:
This directory will house offline and cross-dataset evaluation tools:
- `evaluator.py`: Standardized evaluation pipeline on validation and test sets
- `calibration.py`: Reliability diagrams and Expected Calibration Error (ECE) with Temperature Scaling
- `stability.py`: Real-time flicker rate and state transition latency measurement
- `cross_dataset.py`: Cross-domain benchmark runner (e.g. RAF-DB trained -> AffectNet/Webcam tested)
- `statistical_tests.py`: Bootstrap confidence intervals (95% CI) and paired hypothesis testing

*Note: In accordance with Step 1/Step 2 research scope, no code implementation occurs until research plan and pipeline thresholds are formally approved.*
