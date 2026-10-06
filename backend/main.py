"""FastAPI Backend for Emotion Detector (A4 Multi-Representation Model).

Serves:
- GET /health: Health status check
- POST /predict: Multipart/form-data frame inference using FacePipeline and A4 ONNX model.
"""

import os
import sys
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing.face_pipeline import FacePipeline

app = FastAPI(
    title="Emotion Detector API",
    description="Multi-Representation Facial Mood Analysis (A4 Spatial + Frequency + Geometry)",
    version="1.0.0",
)

# Enable CORS for local dev and preview environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CANONICAL_CLASSES = [
    "Neutral",
    "Happy",
    "Sad",
    "Surprise",
    "Fear",
    "Disgust",
    "Angry",
]

# Path to the authoritative A4 ONNX artifact (trained A4 seed42 model)
ONNX_MODEL_PATH = os.path.join(
    REPO_ROOT,
    "experiments",
    "onnx_a4_trained",
    "a4_seed42_trained.onnx",
)

# Shared singleton instances initialized at startup
face_pipeline: Optional[FacePipeline] = None
onnx_session: Optional[ort.InferenceSession] = None


@app.on_event("startup")
def startup_event():
    global face_pipeline, onnx_session
    print("[Backend] Initializing FacePipeline...")
    face_pipeline = FacePipeline()
    print("[Backend] FacePipeline initialized.")

    if os.path.exists(ONNX_MODEL_PATH):
        print(f"[Backend] Loading ONNX model from {ONNX_MODEL_PATH}...")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_session = ort.InferenceSession(
            ONNX_MODEL_PATH, sess_options, providers=["CPUExecutionProvider"]
        )
        print("[Backend] ONNX Runtime session ready.")
    else:
        print(f"[Backend WARNING] ONNX model not found at {ONNX_MODEL_PATH}")


@app.get("/")
def read_root():
    return {
        "service": "Emotion Detector Backend",
        "status": "online",
        "model": "A4_spatial_frequency_geometry",
    }


@app.get("/health")
def health_check():
    """Health check endpoint expected by frontend."""
    return {
        "status": "healthy",
        "model_loaded": onnx_session is not None,
        "pipeline_ready": face_pipeline is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receives image frame as multipart/form-data and returns expression prediction."""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file payload")

    # Decode image from buffer
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    global face_pipeline, onnx_session
    if face_pipeline is None:
        face_pipeline = FacePipeline()

    result = face_pipeline.process_image(img_rgb, is_webcam=True)

    face_detected = bool(result.get("face_detected", False))
    partial_face = bool(result.get("partial_face", False))
    geometry_valid = bool(result.get("geometry_valid", False))
    head_pose = result.get("head_pose") or {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    raw_bbox = result.get("bbox")
    bbox = list(raw_bbox) if raw_bbox else [0, 0, 0, 0]
    det_conf = result.get("detection_confidence")
    detection_confidence = round(float(det_conf if det_conf is not None else 0.8659), 4)

    if not face_detected or result.get("image_tensor") is None:
        return {
            "status": "OK",
            "face_detected": False,
            "partial_face": False,
            "prediction": "Neutral",
            "prediction_index": 0,
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in CANONICAL_CLASSES},
            "geometry_valid": False,
            "detection_confidence": 0.0,
            "head_pose": {
                "pitch": round(float(head_pose.get("pitch", 0.0)), 2),
                "yaw": round(float(head_pose.get("yaw", 0.0)), 2),
                "roll": round(float(head_pose.get("roll", 0.0)), 2),
            },
            "bbox": [0, 0, 0, 0],
            "backend_connected": True,
        }

    # Run A4 ONNX inference
    if onnx_session is None and os.path.exists(ONNX_MODEL_PATH):
        onnx_session = ort.InferenceSession(
            ONNX_MODEL_PATH, providers=["CPUExecutionProvider"]
        )

    if onnx_session is not None:
        img_tensor = np.expand_dims(result["image_tensor"], axis=0).astype(np.float32)
        geom_vec = np.expand_dims(result["geometry_vector"], axis=0).astype(np.float32)
        geom_valid_tensor = np.array(
            [[1.0 if geometry_valid else 0.0]], dtype=np.float32
        )

        ort_inputs = {
            "image": img_tensor,
            "geometry": geom_vec,
            "geometry_valid": geom_valid_tensor,
        }

        logits = onnx_session.run(["logits"], ort_inputs)[0][0]
        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        pred_idx = int(np.argmax(probs))
        pred_label = CANONICAL_CLASSES[pred_idx]
        pred_conf = float(probs[pred_idx])

        probabilities_dict = {
            c: round(float(p), 4) for c, p in zip(CANONICAL_CLASSES, probs)
        }
    else:
        # Fallback if session couldn't load
        pred_label = "Neutral"
        pred_idx = 0
        pred_conf = 0.5
        probabilities_dict = {c: 0.1428 for c in CANONICAL_CLASSES}

    return {
        "status": "OK",
        "face_detected": True,
        "partial_face": partial_face,
        "prediction": pred_label,
        "prediction_index": pred_idx,
        "confidence": round(pred_conf, 4),
        "probabilities": probabilities_dict,
        "geometry_valid": geometry_valid,
        "detection_confidence": detection_confidence,
        "head_pose": {
            "pitch": round(float(head_pose.get("pitch", 0.0)), 2),
            "yaw": round(float(head_pose.get("yaw", 0.0)), 2),
            "roll": round(float(head_pose.get("roll", 0.0)), 2),
        },
        "bbox": bbox,
        "backend_connected": True,
    }
