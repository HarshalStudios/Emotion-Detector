# Training Module Directory

Scope & Boundary:
This directory will contain the training engine:
- `trainer.py`: Multi-seed training loop with mixed precision (AMP)
- `losses.py`: Class-weighted cross entropy with label smoothing
- `schedulers.py`: Cosine annealing with linear warmup
- `metrics.py`: Macro-F1, balanced accuracy, per-class F1, and ECE tracking
- `early_stopping.py`: Validation Macro-F1 early stopping monitor

*Note: In accordance with Step 1/Step 2 research scope, no code implementation occurs until research plan and pipeline thresholds are formally approved.*
