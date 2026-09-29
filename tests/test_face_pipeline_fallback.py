"""
Unit and Integration Tests for Preprocessing Dataset Fallback and Webcam Invariance.

Verifies:
Test A: Dataset image with successful face detection -> face_detected=True, fallback_used=False, image_tensor present
Test B: Dataset image with simulated zero-face result -> face_detected=False, fallback_used=True, image_tensor shape (3,224,224), geometry_valid=False, geometry_vector all zeros
Test C: Webcam image with zero-face result -> face_detected=False, fallback_used=False, image_tensor=None
Test D: Dataset DataLoader integrity verification -> 0 samples discarded
"""

import os
import sys
import numpy as np
import cv2

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing.face_pipeline import FacePipeline, IMAGENET_MEAN, IMAGENET_STD


def create_synthetic_face_canvas() -> np.ndarray:
    """Generates synthetic portrait image recognizable by face detection."""
    if os.path.exists("test_face.jpg"):
        img = cv2.imread("test_face.jpg")
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    canvas = np.full((300, 300, 3), 200, dtype=np.uint8)
    cv2.ellipse(canvas, (150, 150), (80, 110), 0, 0, 360, (220, 180, 150), -1)
    cv2.circle(canvas, (120, 130), 12, (50, 40, 30), -1)
    cv2.circle(canvas, (180, 130), 12, (50, 40, 30), -1)
    cv2.line(canvas, (150, 140), (150, 175), (140, 100, 80), 4)
    cv2.line(canvas, (120, 200), (180, 200), (120, 50, 50), 5)
    return canvas


def test_a_dataset_successful_detection():
    print("\n--- Test A: Dataset Image with Successful Face Detection ---")
    pipeline = FacePipeline()
    face_img = create_synthetic_face_canvas()

    res = pipeline.process_image(face_img, is_webcam=False)

    print(f"  face_detected: {res['face_detected']}")
    print(f"  fallback_used: {res['fallback_used']}")
    print(f"  image_tensor present: {res['image_tensor'] is not None}")
    if res["image_tensor"] is not None:
        print(f"  image_tensor shape: {res['image_tensor'].shape}, dtype: {res['image_tensor'].dtype}")

    assert res["face_detected"] is True, "Test A FAILED: face_detected should be True"
    assert res["fallback_used"] is False, "Test A FAILED: fallback_used should be False"
    assert res["image_tensor"] is not None, "Test A FAILED: image_tensor should not be None"
    assert res["image_tensor"].shape == (3, 224, 224), "Test A FAILED: image_tensor shape mismatch"
    assert res["geometry_valid"] is True, "Test A FAILED: geometry_valid should be True"
    print("  --> PASS: Test A passed successfully.")


def test_b_dataset_simulated_zero_face():
    print("\n--- Test B: Dataset Image with Simulated Zero-Face Result ---")
    pipeline = FacePipeline()
    # A completely blank canvas will yield zero face landmarks in MediaPipe
    blank_img = np.zeros((100, 100, 3), dtype=np.uint8)

    res = pipeline.process_image(blank_img, is_webcam=False)

    print(f"  face_detected: {res['face_detected']}")
    print(f"  fallback_used: {res['fallback_used']}")
    print(f"  image_tensor present: {res['image_tensor'] is not None}")
    if res["image_tensor"] is not None:
        print(f"  image_tensor shape: {res['image_tensor'].shape}, dtype: {res['image_tensor'].dtype}")
    print(f"  geometry_valid: {res['geometry_valid']}")
    print(f"  geometry_vector is all zeros: {np.all(res['geometry_vector'] == 0.0)}")
    print(f"  bbox: {res['bbox']}")
    print(f"  detection_confidence: {res['detection_confidence']}")

    assert res["face_detected"] is False, "Test B FAILED: face_detected must be False (do not claim face was detected)"
    assert res["fallback_used"] is True, "Test B FAILED: fallback_used must be True"
    assert res["image_tensor"] is not None, "Test B FAILED: image_tensor must be present"
    assert res["image_tensor"].shape == (3, 224, 224), "Test B FAILED: tensor shape must be (3, 224, 224)"
    assert res["image_tensor"].dtype == np.float32, "Test B FAILED: tensor dtype must be float32"
    assert res["geometry_valid"] is False, "Test B FAILED: geometry_valid must be False"
    assert np.all(res["geometry_vector"] == 0.0), "Test B FAILED: geometry_vector must be all zeros"
    assert res["bbox"] is None, "Test B FAILED: bbox must be None"
    assert res["detection_confidence"] is None, "Test B FAILED: detection_confidence must be None"

    # Also verify normalization on blank image: (0 - mean) / std
    expected_channel_0 = (0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    np.testing.assert_allclose(res["image_tensor"][0, 0, 0], expected_channel_0, rtol=1e-5)
    print("  --> PASS: Test B passed successfully.")


def test_c_webcam_zero_face():
    print("\n--- Test C: Webcam Image with Zero-Face Result ---")
    pipeline = FacePipeline()
    blank_img = np.zeros((100, 100, 3), dtype=np.uint8)

    res = pipeline.process_image(blank_img, is_webcam=True)

    print(f"  face_detected: {res['face_detected']}")
    print(f"  fallback_used: {res['fallback_used']}")
    print(f"  image_tensor: {res['image_tensor']}")
    print(f"  geometry_valid: {res['geometry_valid']}")
    print(f"  geometry_vector is all zeros: {np.all(res['geometry_vector'] == 0.0)}")

    assert res["face_detected"] is False, "Test C FAILED: face_detected must be False"
    assert res["fallback_used"] is False, "Test C FAILED: fallback_used must be False on webcam"
    assert res["image_tensor"] is None, "Test C FAILED: image_tensor must be None on webcam no-face"
    assert res["geometry_valid"] is False, "Test C FAILED: geometry_valid must be False"
    assert np.all(res["geometry_vector"] == 0.0), "Test C FAILED: geometry_vector must be all zeros"
    print("  --> PASS: Test C passed successfully (webcam invariance preserved).")


def test_d_no_training_samples_discarded():
    print("\n--- Test D: DataLoader Integrity Verification (Zero Discards) ---")
    pipeline = FacePipeline()

    # Simulate batch of varied images including blank, dark, small, and valid
    samples = [
        np.zeros((100, 100, 3), dtype=np.uint8),                     # zero face
        create_synthetic_face_canvas(),                              # detected face
        np.full((80, 80, 3), 10, dtype=np.uint8),                    # dark square (zero face)
        np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8),    # noise (zero face)
        create_synthetic_face_canvas()                               # detected face
    ]

    tensors = []
    discarded = 0
    for s in samples:
        out = pipeline.process_image(s, is_webcam=False)
        if out["image_tensor"] is not None:
            tensors.append(out["image_tensor"])
        else:
            discarded += 1

    print(f"  Input samples: {len(samples)}")
    print(f"  Successfully extracted tensors: {len(tensors)}")
    print(f"  Discarded count: {discarded}")

    assert len(tensors) == len(samples), "Test D FAILED: Some training samples were discarded!"
    assert discarded == 0, "Test D FAILED: Discarded samples count is non-zero!"
    print("  --> PASS: Test D passed successfully (0 samples discarded).")


if __name__ == "__main__":
    test_a_dataset_successful_detection()
    test_b_dataset_simulated_zero_face()
    test_c_webcam_zero_face()
    test_d_no_training_samples_discarded()
    print("\n=======================================================")
    print("ALL TESTS (A, B, C, D) PASSED WITH FULL SPEC COMPLIANCE!")
    print("=======================================================")
