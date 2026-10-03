#!/usr/bin/env python3
"""scripts/verify_spatial_frequency_batch.py

Pre-training verification script for Experiment A2 (Spatial + Frequency Model).
Performs forward/backward verification on one real batch from the cached RAF-DB dataset,
auditing tensor shapes, finite loss, and gradient backpropagation across all parameter groups.

Usage:
    python scripts/verify_spatial_frequency_batch.py [--cache data/cache/rafdb_cache] [--batch-size 4]
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Verify Spatial+Frequency Model on Real Cached Batch")
    parser.add_argument("--cache", type=str, default="data/cache/rafdb_cache", help="Path to cached RAF-DB directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for verification")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    print("=" * 75)
    print("EXPERIMENT A2: SPATIAL + FREQUENCY BATCH VERIFICATION")
    print(f"Target Cache: {os.path.abspath(args.cache)}")
    print(f"Device:       {args.device}")
    print("=" * 75)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERROR: PyTorch is required to run this verification script.", file=sys.stderr)
        sys.exit(1)

    from src.models.spatial_frequency import create_spatial_frequency_model
    from src.training.dataset import create_cached_rafdb_dataloader

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"Using device: {device}")

    # 1. Instantiate Model
    print("\n1. Instantiating SpatialFrequencyModel (MobileNetV3-Large + SpectralFilter)...")
    torch.manual_seed(42)
    model = create_spatial_frequency_model(
        num_classes=7,
        pretrained=True,
        freq_embedding_dim=256,
        dropout_rate=0.2,
    ).to(device)
    model.train()

    # 2. Count parameters per branch
    spatial_params = sum(p.numel() for p in model.spatial_branch.parameters())
    spatial_trainable = sum(p.numel() for p in model.spatial_branch.parameters() if p.requires_grad)

    freq_params = sum(p.numel() for p in model.frequency_branch.parameters())
    freq_trainable = sum(p.numel() for p in model.frequency_branch.parameters() if p.requires_grad)

    fusion_params = sum(p.numel() for p in model.fusion_head.parameters())
    fusion_trainable = sum(p.numel() for p in model.fusion_head.parameters() if p.requires_grad)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n--- Model Architecture & Parameter Counts ---")
    print(f"  Spatial Branch (MobileNetV3-Large): {spatial_params:>10,} params ({spatial_trainable:,} trainable)")
    print(f"  Frequency Branch (Spectral + Conv): {freq_params:>10,} params ({freq_trainable:,} trainable)")
    print(f"  Fusion & Classification Head:       {fusion_params:>10,} params ({fusion_trainable:,} trainable)")
    print(f"  -------------------------------------------------------------")
    print(f"  Total Model Parameters:             {total_params:>10,} params ({total_trainable:,} trainable)")

    # 3. Load one real batch from cached dataset
    print(f"\n2. Loading one batch (size={args.batch_size}) from {args.cache}...")
    if not os.path.exists(args.cache):
        print(f"ERROR: Cache directory {args.cache} does not exist.", file=sys.stderr)
        sys.exit(1)

    _, loader = create_cached_rafdb_dataloader(
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        cache_dir=args.cache,
    )

    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    print(f"  Loaded images batch shape: {images.shape} (dtype: {images.dtype})")
    print(f"  Loaded labels batch shape: {labels.shape} (values: {labels.cpu().tolist()})")

    # 4. Forward Pass & Feature Extraction
    print("\n3. Executing Forward Pass...")
    features = model.extract_features(images)
    logits = model(images)

    print(f"  Spatial Embedding shape:   {features['spatial_embedding'].shape}")
    print(f"  Frequency Embedding shape: {features['frequency_embedding'].shape}")
    print(f"  Fused Embedding shape:     {features['fused_embedding'].shape}")
    print(f"  Output Logits shape:       {logits.shape}")

    assert logits.shape == (args.batch_size, 7), f"Expected ({args.batch_size}, 7), got {logits.shape}"
    assert torch.isfinite(logits).all(), "Non-finite logits detected!"
    print("  --> Logits shape and finiteness verified: PASS")

    # 5. Loss Computation and Backward Pass
    print("\n4. Executing Loss Computation and Backward Pass...")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss = criterion(logits, labels)
    print(f"  CrossEntropyLoss: {loss.item():.4f}")

    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"

    optimizer = torch.optim.AdamW(model.get_param_groups())
    optimizer.zero_grad()
    loss.backward()

    # Audit gradients
    spatial_grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.spatial_branch.parameters())
    freq_grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.frequency_branch.parameters())
    fusion_grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.fusion_head.parameters())

    print(f"  Spatial Branch Gradients:   {'POPULATED & FINITE (PASS)' if spatial_grad_ok else 'FAILED'}")
    print(f"  Frequency Branch Gradients: {'POPULATED & FINITE (PASS)' if freq_grad_ok else 'FAILED'}")
    print(f"  Fusion Head Gradients:      {'POPULATED & FINITE (PASS)' if fusion_grad_ok else 'FAILED'}")

    assert spatial_grad_ok and freq_grad_ok and fusion_grad_ok, "One or more parameter groups failed gradient check!"

    optimizer.step()
    print("\n" + "=" * 75)
    print("BATCH VERIFICATION 100% SUCCESSFUL: Ready for training on Tesla T4 / GPU.")
    print("=" * 75)


if __name__ == "__main__":
    main()
