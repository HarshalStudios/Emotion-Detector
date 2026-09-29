#!/usr/bin/env python3
"""
scripts/benchmark_gate2_backbones.py

Gate 2 Benchmark: Candidate Backbone Latency and Throughput Evaluation
Measures end-to-end CPU inference latency (mean, P50, P95) and FPS on 224x224 RGB input
across 1 CPU thread and 4 CPU threads using ONNX Runtime CPUExecutionProvider.
"""

import os
import sys
import time
import json
import platform
import psutil
import numpy as np
from datetime import datetime, timezone
import onnxruntime as ort

# Ensure model export directory exists
EXPORT_DIR = "models/onnx_exports"
OUTPUT_JSON = "docs/benchmark_results/gate2_backbone_benchmark.json"

WARMUP_ITERATIONS = 100
TIMED_ITERATIONS = 1000
THREAD_CONFIGS = [1, 4]

MODELS_CONFIG = [
    {
        "name": "MobileNetV3-Large",
        "id": "mobilenet_v3_large",
        "onnx_path": os.path.join(EXPORT_DIR, "mobilenet_v3_large.onnx"),
        "params": "5.48M",
    },
    {
        "name": "EfficientNet-B0",
        "id": "efficientnet_b0",
        "onnx_path": os.path.join(EXPORT_DIR, "efficientnet_b0.onnx"),
        "params": "5.29M",
    },
    {
        "name": "ConvNeXt-Tiny",
        "id": "convnext_tiny",
        "onnx_path": os.path.join(EXPORT_DIR, "convnext_tiny.onnx"),
        "params": "28.59M",
    },
]

def collect_environment_metadata():
    """Collects hardware and software specifications automatically."""
    cpu_model = "Unknown"
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        if cpu_model == "Unknown" or not cpu_model:
            cpu_model = platform.processor() or "x86_64"
    except Exception:
        cpu_model = platform.processor() or "x86_64"

    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or 1

    # Software versions
    import cv2
    import torch

    return {
        "hardware": {
            "cpu": cpu_model,
            "architecture": platform.machine(),
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "ram_gb": ram_gb,
            "os": f"{platform.system()} {platform.release()}",
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "onnxruntime": ort.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        }
    }

def benchmark_model_on_threads(onnx_path: str, num_threads: int, warmup: int = 100, iterations: int = 1000):
    """
    Benchmarks single model on ONNX Runtime CPUExecutionProvider with specified thread count.
    Measures end-to-end model inference only.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Standard 224x224 RGB input tensor (batch size = 1)
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warmup phase (100 iterations)
    for _ in range(warmup):
        _ = session.run([output_name], {input_name: dummy_input})

    # Timed benchmark phase (1000 iterations)
    latencies_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(latencies_arr))
    p50_ms = float(np.percentile(latencies_arr, 50))
    p95_ms = float(np.percentile(latencies_arr, 95))
    fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return {
        "thread_count": num_threads,
        "warmup_count": warmup,
        "timed_iteration_count": iterations,
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(p50_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "fps": round(fps, 1),
    }

def run_gate2_benchmark():
    print("=" * 78)
    print("GATE 2 CANDIDATE BACKBONE INFERENCE BENCHMARK")
    print("Standardized ONNX Runtime CPUExecutionProvider (Batch = 1, 224x224 RGB)")
    print("=" * 78)

    env_meta = collect_environment_metadata()
    print("\n[Environment Metadata]")
    print(f"  CPU: {env_meta['hardware']['cpu']} ({env_meta['hardware']['architecture']})")
    print(f"  Cores: {env_meta['hardware']['physical_cores']} Physical / {env_meta['hardware']['logical_cores']} Logical")
    print(f"  RAM: {env_meta['hardware']['ram_gb']} GB")
    print(f"  OS: {env_meta['hardware']['os']}")
    print(f"  Software: Python {env_meta['software']['python']} | PyTorch {env_meta['software']['pytorch']} | ONNX Runtime {env_meta['software']['onnxruntime']} | OpenCV {env_meta['software']['opencv']}")

    # Verify model files exist
    for m in MODELS_CONFIG:
        if not os.path.exists(m["onnx_path"]):
            raise FileNotFoundError(f"Model file missing: {m['onnx_path']}")
        file_size_mb = os.path.getsize(m["onnx_path"]) / (1024 * 1024)
        print(f"  Loaded model: {m['name']} ({m['onnx_path']}) -> {file_size_mb:.2f} MB")

    timestamp = datetime.now(timezone.utc).isoformat()
    results = []

    for m in MODELS_CONFIG:
        print(f"\n>>> Benchmarking Backbone: {m['name']} (Params: {m['params']})")
        model_results = {
            "model_name": m["name"],
            "model_id": m["id"],
            "parameters": m["params"],
            "onnx_size_bytes": os.path.getsize(m["onnx_path"]),
            "onnx_size_mb": round(os.path.getsize(m["onnx_path"]) / (1024 * 1024), 2),
            "thread_benchmarks": {}
        }

        for threads in THREAD_CONFIGS:
            print(f"  Running on {threads} CPU thread(s) ({WARMUP_ITERATIONS} warmup, {TIMED_ITERATIONS} timed iterations)...", end="", flush=True)
            bench = benchmark_model_on_threads(
                m["onnx_path"],
                num_threads=threads,
                warmup=WARMUP_ITERATIONS,
                iterations=TIMED_ITERATIONS
            )
            print(f" Done! Mean: {bench['mean_ms']:.2f}ms | P50: {bench['p50_ms']:.2f}ms | P95: {bench['p95_ms']:.2f}ms | {bench['fps']:.1f} FPS")
            model_results["thread_benchmarks"][f"{threads}_thread"] = bench

        results.append(model_results)

    # Save to JSON
    output_data = {
        "benchmark_name": "Gate_2_Backbone_CPU_Latency_Benchmark",
        "timestamp": timestamp,
        "methodology": {
            "framework": "ONNX Runtime CPUExecutionProvider",
            "input_resolution": [3, 224, 224],
            "batch_size": 1,
            "warmup_iterations": WARMUP_ITERATIONS,
            "timed_iterations": TIMED_ITERATIONS,
            "timing_function": "time.perf_counter",
            "threads_evaluated": THREAD_CONFIGS,
            "metrics": ["mean_ms", "p50_ms", "p95_ms", "fps (1000 / mean_ms)"]
        },
        "hardware": env_meta["hardware"],
        "software": env_meta["software"],
        "models": results,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Saved Artifact]: Benchmark successfully recorded at {OUTPUT_JSON}")
    print("=" * 78)
    return output_data

if __name__ == "__main__":
    run_gate2_benchmark()
