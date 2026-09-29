# Configuration Files Directory

This directory stores reproducible YAML configuration files defining all experimental parameters:
- Architecture specifications (backbone, branch dimensions, gating configuration)
- Hyperparameters (learning rates, weight decays, batch sizes, warmup steps, schedulers)
- Loss weights and label smoothing
- Preprocessing and augmentation parameters
- Random seeds and dataset paths

Every training run must reference a frozen configuration file from this folder.
