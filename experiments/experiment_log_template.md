# Experiment Log Template

| Column Name | Description | Example |
|---|---|---|
| `exp_id` | Unique identifier formatted as `EXP-###-variant-seed` | `EXP-001-A0_baseline-s42` |
| `date` | Timestamp in YYYY-MM-DD format | `2026-10-05` |
| `git_commit` | Git short SHA of the codebase when experiment was run | `a1b2c3d` |
| `config_path` | Path to the exact configuration file | `configs/baselines/convnext_tiny_s42.yaml` |
| `model_architecture` | Full architecture tag (Backbone + active branches + fusion type) | `ConvNeXt-Tiny (Spatial only)` |
| `dataset_train` | Dataset name, split, and class count | `RAF-DB basic (train split, 12,271 imgs)` |
| `dataset_val` | Validation dataset and split used for early stopping | `RAF-DB basic (val split, 3,068 imgs)` |
| `input_resolution` | Input image tensor dimensions `(C x H x W)` | `3 x 224 x 224` |
| `active_branches` | Comma-separated list of branches enabled | `spatial` |
| `augmentation` | Augmentation pipeline description | `RandomHorizontalFlip, ColorJitter(0.2), Affine(scale=0.1, rot=10)` |
| `optimizer` | Optimizer and weight decay | `AdamW (weight_decay=1e-2)` |
| `lr_backbone` | Learning rate for pretrained backbone weights | `1.0e-4` |
| `lr_heads` | Learning rate for newly initialized heads / branches | `1.0e-3` |
| `lr_scheduler` | Scheduler type and warmup epochs | `CosineAnnealingLR (T_max=30, warmup=3)` |
| `batch_size` | Effective batch size | `32` |
| `epochs_max` | Maximum epochs scheduled | `30` |
| `epochs_trained` | Actual epochs completed before early stopping | `22` |
| `loss_function` | Loss configuration | `CrossEntropyLoss(label_smoothing=0.1, class_weights=balanced)` |
| `seed` | Random seed integer | `42` |
| `val_acc` | Overall validation top-1 accuracy (%) | `88.42%` |
| `val_balanced_acc` | Mean per-class recall / balanced accuracy (%) | `84.15%` |
| `val_macro_f1` | Primary metric: Macro-averaged F1 score (0.0 to 1.0) | `0.8431` |
| `val_per_class_f1` | JSON string or array of per-class F1 `[neu, hap, sad, sur, fea, dis, ang]` | `[0.86, 0.93, 0.81, 0.84, 0.65, 0.62, 0.79]` |
| `val_ece_15bins` | Expected Calibration Error across 15 equal bins | `0.0482` |
| `cpu_lat_p50_ms` | ONNX Runtime median single-face latency on target CPU (batch=1) | `18.4 ms` |
| `cpu_lat_p95_ms` | 95th percentile latency on target CPU | `24.1 ms` |
| `fps_single_thread`| Measured inference throughput (FPS) | `51.2 FPS` |
| `param_count_m` | Total trainable parameters in millions | `28.6 M` |
| `checkpoint_path` | Absolute or relative path to saved model weights | `checkpoints/EXP-001/best_macro_f1.pt` |
| `notes_and_failures`| Observations, edge cases, anomalies, failure modes | `Slight drop in Disgust F1 during epoch 18; converged smoothly` |

## CSV Header String
```csv
exp_id,date,git_commit,config_path,model_architecture,dataset_train,dataset_val,input_resolution,active_branches,augmentation,optimizer,lr_backbone,lr_heads,lr_scheduler,batch_size,epochs_max,epochs_trained,loss_function,seed,val_acc,val_balanced_acc,val_macro_f1,val_per_class_f1,val_ece_15bins,cpu_lat_p50_ms,cpu_lat_p95_ms,fps_single_thread,param_count_m,checkpoint_path,notes_and_failures
```
