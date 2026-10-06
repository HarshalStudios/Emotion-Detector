import os
import torch

ckpt_path = "model_artifacts/a4_seed42/a4_best_model.pt"
print(f"File exists: {os.path.exists(ckpt_path)}")
print(f"File size: {os.path.getsize(ckpt_path)} bytes")

checkpoint = torch.load(ckpt_path, map_location="cpu")
print("Top-level keys:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint))

if isinstance(checkpoint, dict):
    for k in checkpoint.keys():
        if k != "model_state_dict":
            print(f"Key '{k}': {checkpoint[k]}")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"Number of tensors in model_state_dict: {len(state_dict)}")
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"Total parameter count in state_dict: {total_params:,}")
        
        # Check first few and last few keys
        keys = list(state_dict.keys())
        print("First 5 keys:", keys[:5])
        print("Last 5 keys:", keys[-5:])
        
        # Check weight statistics
        for sample_key in [keys[0], keys[len(keys)//2], keys[-1]]:
            t = state_dict[sample_key].float()
            print(f"Weight '{sample_key}': shape={t.shape}, mean={t.mean().item():.6f}, std={t.std().item():.6f}, min={t.min().item():.6f}, max={t.max().item():.6f}")
