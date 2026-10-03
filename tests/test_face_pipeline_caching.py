"""
Tests for Cache Readiness: Exposing pre-normalization uint8 RGB image,
portable MediaPipe model paths, consistency verification, and fallback behavior.
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Add repository root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.face_pipeline import FacePipeline, IMAGENET_MEAN, IMAGENET_STD


def get_test_face() -> np.ndarray:
    """Returns a real test face if available, otherwise a synthetic face canvas."""
    face_path = os.path.join(PROJECT_ROOT, "test_face.jpg")
    if os.path.exists(face_path):
        img_bgr = cv2.imread(face_path)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    canvas = np.full((320, 320, 3), 220, dtype=np.uint8)
    cv2.ellipse(canvas, (160, 160), (90, 120), 0, 0, 360, (210, 180, 150), -1)
    cv2.circle(canvas, (130, 140), 12, (50, 40, 30), -1)
    cv2.circle(canvas, (190, 140), 12, (50, 40, 30), -1)
    cv2.line(canvas, (160, 150), (160, 185), (140, 100, 80), 4)
    cv2.line(canvas, (130, 210), (190, 210), (120, 50, 50), 5)
    return canvas


class TestFacePipelineCachingReadiness(unittest.TestCase):

    def setUp(self):
        self.pipeline = FacePipeline()
        self.test_face_rgb = get_test_face()

    def test_01_successful_detection_uint8_and_tensor_shapes(self):
        """Verify uint8 shape, dtype, tensor shape, dtype, and backward compatibility."""
        res = self.pipeline.process_image(self.test_face_rgb, is_webcam=False)

        self.assertTrue(res["face_detected"], "Expected face to be detected.")
        self.assertFalse(res["fallback_used"], "Expected fallback_used to be False on valid face.")

        # 1. Verify aligned_image_rgb properties
        self.assertIn("aligned_image_rgb", res, "Key 'aligned_image_rgb' must be present in output.")
        aligned_rgb = res["aligned_image_rgb"]
        self.assertIsNotNone(aligned_rgb, "aligned_image_rgb must not be None on successful detection.")
        self.assertIsInstance(aligned_rgb, np.ndarray, "aligned_image_rgb must be a numpy ndarray.")
        self.assertEqual(aligned_rgb.shape, (224, 224, 3), f"Expected shape (224, 224, 3), got {aligned_rgb.shape}")
        self.assertEqual(aligned_rgb.dtype, np.uint8, f"Expected dtype uint8, got {aligned_rgb.dtype}")

        # 2. Verify image_tensor properties
        self.assertIn("image_tensor", res, "Key 'image_tensor' must be present in output.")
        tensor = res["image_tensor"]
        self.assertIsNotNone(tensor, "image_tensor must not be None.")
        self.assertEqual(tensor.shape, (3, 224, 224), f"Expected shape (3, 224, 224), got {tensor.shape}")
        self.assertEqual(tensor.dtype, np.float32, f"Expected dtype float32, got {tensor.dtype}")

        # 3. Geometry vector
        self.assertTrue(res["geometry_valid"], "Expected geometry_valid to be True.")
        self.assertEqual(res["geometry_vector"].shape, (62,), "Expected 62-D geometry vector.")

    def test_02_uint8_rgb_and_normalized_tensor_mathematical_consistency(self):
        """Verify that normalized image_tensor is mathematically identical to normalized aligned_image_rgb."""
        res = self.pipeline.process_image(self.test_face_rgb, is_webcam=False)
        aligned_rgb = res["aligned_image_rgb"]
        tensor = res["image_tensor"]

        # Re-apply ImageNet normalization formula to uint8 RGB
        expected_tensor_hwc = (aligned_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        expected_tensor_chw = np.transpose(expected_tensor_hwc, (2, 0, 1)).astype(np.float32)

        # Check maximum absolute difference is negligible
        max_diff = float(np.max(np.abs(tensor - expected_tensor_chw)))
        self.assertLess(
            max_diff,
            1e-5,
            f"Mathematical inconsistency between aligned_image_rgb and image_tensor! Max diff: {max_diff}"
        )

    def test_03_dataset_mode_no_face_fallback(self):
        """Verify no-face dataset fallback returns usable 224x224 uint8, 224x224 tensor, geometry_valid=False, zero geometry."""
        blank_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        res = self.pipeline.process_image(blank_rgb, is_webcam=False)

        self.assertFalse(res["face_detected"], "face_detected must be False for zero-face input.")
        self.assertTrue(res["fallback_used"], "fallback_used must be True in dataset mode.")
        self.assertFalse(res["geometry_valid"], "geometry_valid must be False in fallback.")
        self.assertTrue(np.all(res["geometry_vector"] == 0.0), "geometry_vector must be all zeros in fallback.")
        self.assertEqual(res["geometry_vector"].shape, (62,))

        # Verify aligned_image_rgb in fallback
        self.assertIn("aligned_image_rgb", res)
        aligned_rgb = res["aligned_image_rgb"]
        self.assertIsNotNone(aligned_rgb)
        self.assertEqual(aligned_rgb.shape, (224, 224, 3))
        self.assertEqual(aligned_rgb.dtype, np.uint8)

        # Verify image_tensor in fallback
        tensor = res["image_tensor"]
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)

        # Consistency in fallback
        expected_chw = np.transpose(
            (aligned_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD,
            (2, 0, 1)
        ).astype(np.float32)
        np.testing.assert_allclose(tensor, expected_chw, rtol=1e-5, atol=1e-5)

    def test_04_webcam_mode_zero_face_invariance(self):
        """Verify webcam mode returns None for both image_tensor and aligned_image_rgb when no face detected."""
        blank_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        res = self.pipeline.process_image(blank_rgb, is_webcam=True)

        self.assertFalse(res["face_detected"])
        self.assertFalse(res["fallback_used"])
        self.assertIsNone(res["image_tensor"])
        self.assertIsNone(res["aligned_image_rgb"])
        self.assertFalse(res["geometry_valid"])
        self.assertTrue(np.all(res["geometry_vector"] == 0.0))

    def test_05_portable_model_paths_resolution(self):
        """Verify model paths resolve without hardcoded /app/applet."""
        # 1. Default initialization should locate files from repo root
        p = FacePipeline()
        self.assertTrue(os.path.exists(p.model_path), f"Resolved model path does not exist: {p.model_path}")
        self.assertTrue(os.path.exists(p.detector_model_path), f"Resolved detector path does not exist: {p.detector_model_path}")
        self.assertTrue(p.model_path.endswith("models/mediapipe/face_landmarker.task"))
        self.assertTrue(p.detector_model_path.endswith("models/mediapipe/blaze_face_short_range.tflite"))

        # 2. Explicit valid path
        explicit_path = os.path.join(PROJECT_ROOT, "models/mediapipe/face_landmarker.task")
        p2 = FacePipeline(model_path=explicit_path)
        self.assertEqual(p2.model_path, os.path.abspath(explicit_path))

        # 3. Nonexistent path raises FileNotFoundError with informative message
        with self.assertRaises(FileNotFoundError) as ctx:
            FacePipeline._resolve_model_path(
                explicit_path="/nonexistent/path/model.task",
                env_var="NONEXISTENT_ENV_VAR",
                rel_filename="nonexistent/relative/file.task",
                model_name="Test Model",
            )
        self.assertIn("Required Test Model file could not be found", str(ctx.exception))

    def test_06_preprocessing_parity_path_a_vs_path_b(self):
        """Verify numerical parity between Path A (Dataset mode) and Path B (Webcam mode) on a valid face."""
        res_a = self.pipeline.process_image(self.test_face_rgb, is_webcam=False)
        res_b = self.pipeline.process_image(self.test_face_rgb, is_webcam=True)

        self.assertTrue(res_a["face_detected"])
        self.assertTrue(res_b["face_detected"])

        # Compare image tensors
        tensor_diff = float(np.max(np.abs(res_a["image_tensor"] - res_b["image_tensor"])))
        self.assertEqual(tensor_diff, 0.0, f"Tensor parity mismatch: {tensor_diff}")

        # Compare uint8 images
        uint8_diff = int(np.max(np.abs(res_a["aligned_image_rgb"].astype(int) - res_b["aligned_image_rgb"].astype(int))))
        self.assertEqual(uint8_diff, 0, f"uint8 image parity mismatch: {uint8_diff}")

        # Compare geometry vectors
        geom_diff = float(np.max(np.abs(res_a["geometry_vector"] - res_b["geometry_vector"])))
        self.assertEqual(geom_diff, 0.0, f"Geometry parity mismatch: {geom_diff}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
