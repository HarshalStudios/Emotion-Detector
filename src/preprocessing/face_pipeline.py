"""
Face Preprocessing Pipeline Module
Strictly implements the locked Step 1 & 2 Specification with full API honesty:
1. MediaPipe Face Landmarker detection (with multi-face largest area rule)
2. 5-point roll-based rigid rotation alignment (rotates face to horizontal inter-ocular axis)
3. 1.30 Margin Crop (symmetric square bounding box expansion around aligned face)
4. 224x224 Bilinear Resize
5. ImageNet Normalization -> (3, 224, 224) float32 tensor
6. 62-D Geometry Representation (52 FACS blendshapes + 10 normalized distance ratios)
7. Webcam-only partial face checking using exact available API measurements:
   - Bounding box width < 64 px or height < 64 px
   - Absolute head pose angles: |yaw| > 45 deg, |pitch| > 35 deg, |roll| > 45 deg
   - NOTE ON CONFIDENCE: MediaPipe Face Landmarker Tasks API does NOT expose a numeric
     per-face confidence score in FaceLandmarkerResult. Detection acceptance is enforced
     by internal graph threshold (min_face_detection_confidence=0.40). The unusable
     confidence field is explicitly set to None and omitted from boolean threshold checks.
8. Training images NEVER discard samples (partial-face check is strictly bypassed).
"""

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure local packages in pylib or site-packages are discovered
if os.path.exists("/app/applet/pylib") and "/app/applet/pylib" not in sys.path:
    sys.path.insert(0, "/app/applet/pylib")

import cv2
import numpy as np

# Canonical 52 MediaPipe Face Blendshape Category Names (alphabetical order)
CANONICAL_BLENDSHAPE_NAMES: List[str] = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward",
    "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
]

# Exact Landmark IDs for 10 Normalized Distance Ratios
# Outer inter-ocular normalization landmarks: 33 (right outer), 263 (left outer)
RATIO_LANDMARK_PAIRS: List[Tuple[int, int]] = [
    (13, 14),    # R1: Lip vertical aperture (inner upper lip center to inner lower lip center)
    (61, 291),   # R2: Mouth horizontal width (right corner to left corner)
    (386, 374),  # R3: Left eye vertical aperture (upper to lower eyelid)
    (159, 145),  # R4: Right eye vertical aperture (upper to lower eyelid)
    (296, 386),  # R5: Left eyebrow raise (eyebrow arch to upper eyelid)
    (66, 159),   # R6: Right eyebrow raise (eyebrow arch to upper eyelid)
    (107, 336),  # R7: Inner brow furrow distance (right inner to left inner brow)
    (1, 152),    # R8: Lower face height (nose tip to chin bottom)
    (291, 279),  # R9: Left nasolabial pull (mouth corner to nose alar corner)
    (61, 49),    # R10: Right nasolabial pull (mouth corner to nose alar corner)
]

# ImageNet standard normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes pitch, yaw, and roll in degrees from a 3x3 rotation matrix.
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return math.degrees(x), math.degrees(y), math.degrees(z)


class FacePipeline:
    """
    Unified face detection, alignment, cropping, and feature extraction pipeline.
    Ensures identical tensor generation across training, testing, and real-time inference.
    """

    @staticmethod
    def _resolve_model_path(
        explicit_path: Optional[str],
        env_var: str,
        rel_filename: str,
        model_name: str,
    ) -> str:
        """
        Resolves model file path portably:
        1. Explicit path if supplied and exists on disk.
        2. Environment variable if set and exists on disk.
        3. Relative to repository root (anchored via this source file's location).
        4. Relative to current working directory.
        5. Legacy /app/applet container path.
        Fails with a clear FileNotFoundError if not found.
        """
        candidates: List[str] = []

        if explicit_path:
            norm_explicit = os.path.abspath(explicit_path)
            candidates.append(norm_explicit)
            if os.path.exists(explicit_path):
                return norm_explicit

        if env_var in os.environ and os.environ[env_var]:
            norm_env = os.path.abspath(os.environ[env_var])
            candidates.append(norm_env)
            if os.path.exists(os.environ[env_var]):
                return norm_env

        # Candidate relative to repository root (src/preprocessing/../.. -> repo root)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        repo_candidate = os.path.join(repo_root, rel_filename)
        candidates.append(repo_candidate)
        if os.path.exists(repo_candidate):
            return repo_candidate

        # Candidate relative to current working directory
        cwd_candidate = os.path.abspath(rel_filename)
        if cwd_candidate not in candidates:
            candidates.append(cwd_candidate)
        if os.path.exists(cwd_candidate):
            return cwd_candidate

        # Candidate in legacy /app/applet container
        legacy_candidate = os.path.join("/app/applet", rel_filename)
        if legacy_candidate not in candidates:
            candidates.append(legacy_candidate)
        if os.path.exists(legacy_candidate):
            return legacy_candidate

        checked = "\n  - " + "\n  - ".join(candidates)
        raise FileNotFoundError(
            f"Required {model_name} file could not be found.\n"
            f"Checked locations:{checked}\n"
            f"Please ensure '{rel_filename}' is present in the repository or specify a valid path via argument or {env_var}."
        )

    def __init__(
        self,
        model_path: Optional[str] = None,
        detector_model_path: Optional[str] = None,
        target_size: int = 224,
        margin_factor: float = 1.30,
    ):
        self.target_size = target_size
        self.margin_factor = margin_factor
        self.model_path = self._resolve_model_path(
            model_path,
            env_var="MEDIAPIPE_LANDMARKER_PATH",
            rel_filename="models/mediapipe/face_landmarker.task",
            model_name="MediaPipe Face Landmarker (.task)",
        )
        self.detector_model_path = self._resolve_model_path(
            detector_model_path,
            env_var="MEDIAPIPE_DETECTOR_PATH",
            rel_filename="models/mediapipe/blaze_face_short_range.tflite",
            model_name="MediaPipe Face Detector (.tflite)",
        )
        self._landmarker = None
        self._detector = None

        if self.model_path and os.path.exists(self.model_path):
            self._init_landmarker(self.model_path)

        if self.detector_model_path and os.path.exists(self.detector_model_path):
            self._init_detector(self.detector_model_path)

    def _init_detector(self, model_path: str) -> None:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.10,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def _init_landmarker(self, model_path: str) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=5,  # Multiple face candidates for largest face rule
            min_face_detection_confidence=0.40,
            min_face_presence_confidence=0.40,
            min_tracking_confidence=0.40,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def _extract_5_points(self, landmarks_px: np.ndarray) -> np.ndarray:
        """
        Extracts 5 canonical reference points from 478 MediaPipe landmark coordinates:
        1. Right eye center (midpoint of outer 33 and inner 133)
        2. Left eye center (midpoint of outer 263 and inner 362)
        3. Nose tip (Landmark 1)
        4. Right mouth corner (Landmark 61)
        5. Left mouth corner (Landmark 291)
        """
        right_eye = (landmarks_px[33, :2] + landmarks_px[133, :2]) / 2.0
        left_eye = (landmarks_px[263, :2] + landmarks_px[362, :2]) / 2.0
        nose_tip = landmarks_px[1, :2]
        right_mouth = landmarks_px[61, :2]
        left_mouth = landmarks_px[291, :2]

        return np.array([right_eye, left_eye, nose_tip, right_mouth, left_mouth], dtype=np.float32)

    def _align_and_margin_crop_sequence(
        self,
        image_rgb: np.ndarray,
        landmarks_px: np.ndarray,
        src_5pts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Explicit production sequence:
        Stage 1: 5-point roll-based rigid rotation alignment (rotates face to horizontal inter-ocular axis)
        Stage 2: 1.30 margin crop around aligned face bounding box
        Stage 3: 224x224 bilinear resize
        Stage 4: ImageNet normalization
        
        Returns:
            normalized_tensor: (3, 224, 224) float32
            aligned_crop_rgb: (224, 224, 3) uint8 image before normalization
            audit_metadata: dict tracking exact intermediate stages for verification
        """
        h_img, w_img, _ = image_rgb.shape
        right_eye = src_5pts[0]
        left_eye = src_5pts[1]

        # Stage 1: 5-point roll-based rigid rotation alignment (Euclidean rigid rotation)
        # Compute roll angle to make inter-ocular line horizontal
        dY = float(left_eye[1] - right_eye[1])
        dX = float(left_eye[0] - right_eye[0])
        roll_angle_deg = math.degrees(math.atan2(dY, dX))

        # Centroid of 5 canonical points is the rotation pivot
        pivot = np.mean(src_5pts, axis=0)
        pivot_tuple = (float(pivot[0]), float(pivot[1]))

        # Construct 2D affine rotation matrix (5-point roll-based rigid rotation with scale 1.0)
        M_rot = cv2.getRotationMatrix2D(pivot_tuple, roll_angle_deg, scale=1.0)

        # Warp full image to aligned horizontal orientation
        aligned_full = cv2.warpAffine(
            image_rgb, M_rot, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

        # Transform landmarks into aligned coordinates
        ones = np.ones((len(landmarks_px), 1), dtype=np.float32)
        pts_homo = np.hstack([landmarks_px[:, :2], ones])
        aligned_pts = (M_rot @ pts_homo.T).T  # (478, 2)

        # Stage 2: 1.30 Margin Crop on Aligned Coordinates
        xs = aligned_pts[:, 0]
        ys = aligned_pts[:, 1]
        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        min_y, max_y = float(np.min(ys)), float(np.max(ys))
        aligned_w = max_x - min_x
        aligned_h = max_y - min_y
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        # Symmetrically expand with 1.30 margin factor
        side = max(aligned_w, aligned_h) * self.margin_factor
        half_side = side / 2.0

        x1 = int(round(cx - half_side))
        y1 = int(round(cy - half_side))
        x2 = int(round(cx + half_side))
        y2 = int(round(cy + half_side))

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w_img)
        pad_bottom = max(0, y2 - h_img)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padded = cv2.copyMakeBorder(
                aligned_full, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
            )
            crop = padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
        else:
            crop = aligned_full[y1:y2, x1:x2]

        # Stage 3: 224x224 Bilinear Resize
        resized_rgb = cv2.resize(
            crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR
        )

        # Stage 4: ImageNet Normalization
        norm_img = resized_rgb.astype(np.float32) / 255.0
        norm_img = (norm_img - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = np.transpose(norm_img, (2, 0, 1)).astype(np.float32)

        audit_meta = {
            "stage_1_alignment_roll_deg": roll_angle_deg,
            "stage_2_margin_side_px": side,
            "stage_2_crop_bbox": (x1, y1, x2 - x1, y2 - y1),
            "stage_3_target_size": (self.target_size, self.target_size),
            "stage_4_tensor_shape": image_tensor.shape,
        }

        return image_tensor, resized_rgb, audit_meta

    def _compute_62d_geometry(
        self,
        blendshapes: List[Any],
        landmarks: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Computes the locked 62-D geometry vector:
        52 FACS Blendshapes + 10 Normalized Distance Ratios.
        """
        if len(landmarks) < 468:
            return np.zeros(62, dtype=np.float32), False

        # 1. 52 FACS Blendshapes (alphabetical order)
        blendshape_dict = {b.category_name: float(b.score) for b in blendshapes}
        vec_52 = np.array([blendshape_dict.get(name, 0.0) for name in CANONICAL_BLENDSHAPE_NAMES], dtype=np.float32)

        # 2. Outer Inter-Ocular Normalizer (Landmark 33 to Landmark 263)
        p33 = landmarks[33]
        p263 = landmarks[263]
        d_norm = float(np.linalg.norm(p33 - p263))
        if d_norm < 1e-5:
            d_norm = 1.0

        # 3. 10 Distance Ratios
        vec_10 = np.zeros(10, dtype=np.float32)
        for idx, (id_a, id_b) in enumerate(RATIO_LANDMARK_PAIRS):
            pa = landmarks[id_a]
            pb = landmarks[id_b]
            dist = float(np.linalg.norm(pa - pb))
            vec_10[idx] = dist / d_norm

        full_vec = np.concatenate([vec_52, vec_10], axis=0).astype(np.float32)
        return full_vec, True

    def process_image(
        self,
        image_rgb: np.ndarray,
        is_webcam: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes complete preprocessing on an RGB image.

        Args:
            image_rgb: Input RGB image numpy array (H, W, 3), uint8.
            is_webcam: Flag indicating live webcam input.
                       When False (training/dataset images): Partial-face checks are DISABLED
                       and samples are NEVER discarded.

        Returns:
            Dict containing:
                face_detected: bool (True only if actual MediaPipe face detected)
                fallback_used: bool (True if dataset fallback was required when zero faces detected)
                partial_face: bool (True only on webcam if conditions violated)
                image_tensor: Optional[np.ndarray] of shape (3, 224, 224), float32 ImageNet normalized
                aligned_image_rgb: Optional[np.ndarray] of shape (224, 224, 3), uint8 pre-normalization crop
                geometry_vector: np.ndarray of shape (62,), float32 (zero-masked if partial/invalid/fallback)
                geometry_valid: bool (1 if geometry is active, 0 if masked/fallback)
                head_pose: Dict[str, float] with pitch, yaw, roll in degrees
                detection_confidence: Optional[float] (real numeric score from MediaPipe FaceDetector)
                bbox: Optional[Tuple[int, int, int, int]] (x, y, w, h)
                audit_meta: Optional[Dict] with stage-by-stage audit data
        """
        img_h, img_w, _ = image_rgb.shape

        if self._landmarker is None:
            return self._emergency_fallback(image_rgb)

        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self._landmarker.detect(mp_image)

        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            if not is_webcam:
                # Dataset fallback mode (is_webcam=False):
                # When MediaPipe detects zero faces in dataset images (e.g. pre-cropped RAF-DB chips),
                # DO NOT discard the sample.
                # DO NOT invent landmarks or a bounding box.
                # DO NOT claim that a face was detected.
                # Resize input RGB directly to 224x224 bilinear, apply ImageNet normalization.
                resized_rgb = cv2.resize(
                    image_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR
                )
                norm_img = resized_rgb.astype(np.float32) / 255.0
                norm_img = (norm_img - IMAGENET_MEAN) / IMAGENET_STD
                image_tensor = np.transpose(norm_img, (2, 0, 1)).astype(np.float32)

                return {
                    "face_detected": False,
                    "fallback_used": True,
                    "partial_face": False,
                    "image_tensor": image_tensor,
                    "aligned_image_rgb": resized_rgb,
                    "geometry_vector": np.zeros(62, dtype=np.float32),
                    "geometry_valid": False,
                    "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                    "detection_confidence": None,
                    "bbox": None,
                    "audit_meta": {
                        "dataset_fallback": True,
                        "stage_3_target_size": (self.target_size, self.target_size),
                        "stage_4_tensor_shape": image_tensor.shape,
                    },
                }
            else:
                # Webcam mode (is_webcam=True):
                # Keep existing no-face behavior strictly unchanged:
                # face_detected=False, fallback_used=False, image_tensor=None
                return {
                    "face_detected": False,
                    "fallback_used": False,
                    "partial_face": False,
                    "image_tensor": None,
                    "aligned_image_rgb": None,
                    "geometry_vector": np.zeros(62, dtype=np.float32),
                    "geometry_valid": False,
                    "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                    "detection_confidence": None,
                    "bbox": None,
                    "audit_meta": None,
                }

        # Multiple Faces Rule: Select face with largest 2D bounding box area
        best_face_idx = 0
        max_area = -1.0
        best_bbox = (0, 0, 0, 0)
        best_pts_px = None

        for face_idx, landmarks in enumerate(detection_result.face_landmarks):
            xs = [lm.x * img_w for lm in landmarks]
            ys = [lm.y * img_h for lm in landmarks]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w = x_max - x_min
            h = y_max - y_min
            area = w * h
            if area > max_area:
                max_area = area
                best_face_idx = face_idx
                best_bbox = (int(x_min), int(y_min), int(w), int(h))
                best_pts_px = np.column_stack([xs, ys, [lm.z * img_w for lm in landmarks]])

        # Head Pose extraction from 4x4 rigid transformation matrix
        pitch, yaw, roll = 0.0, 0.0, 0.0
        if detection_result.facial_transformation_matrixes and len(detection_result.facial_transformation_matrixes) > best_face_idx:
            mat4 = np.array(detection_result.facial_transformation_matrixes[best_face_idx])
            R = mat4[:3, :3]
            pitch, yaw, roll = rotation_matrix_to_euler_angles(R)

        # MediaPipe Face Detector: Obtain real detection confidence
        detection_confidence = None
        if self._detector is not None:
            det_result = self._detector.detect(mp_image)
            if det_result.detections and len(det_result.detections) > 0:
                # Multiple Faces Rule: Select face with largest 2D bounding box area
                best_det = max(
                    det_result.detections,
                    key=lambda d: d.bounding_box.width * d.bounding_box.height,
                )
                if best_det.categories and len(best_det.categories) > 0:
                    detection_confidence = float(best_det.categories[0].score)

        # Partial-Face Validation Rule (Strictly Webcam Only)
        partial_face = False
        partial_reason = None
        if is_webcam:
            bbox_w, bbox_h = best_bbox[2], best_bbox[3]
            if bbox_w < 64 or bbox_h < 64:
                partial_face = True
                partial_reason = f"Face dimensions below 64px: {bbox_w}x{bbox_h}"
            elif detection_confidence is not None and detection_confidence < 0.50:
                partial_face = True
                partial_reason = f"Detection confidence below 0.50: {detection_confidence:.3f} < 0.50"
            elif abs(yaw) > 45.0:
                partial_face = True
                partial_reason = f"Extreme yaw: {yaw:.1f} deg > 45 deg"
            elif abs(pitch) > 35.0:
                partial_face = True
                partial_reason = f"Extreme pitch: {pitch:.1f} deg > 35 deg"
            elif abs(roll) > 45.0:
                partial_face = True
                partial_reason = f"Extreme roll: {roll:.1f} deg > 45 deg"

        # Production Execution Sequence:
        # 5-point alignment -> 1.30 margin crop -> 224x224 resize -> ImageNet normalization
        src_5pts = self._extract_5_points(best_pts_px)
        image_tensor, aligned_crop_rgb, audit_meta = self._align_and_margin_crop_sequence(
            image_rgb, best_pts_px, src_5pts
        )
        audit_meta["partial_reason"] = partial_reason

        # 62-D Geometry Representation
        blendshapes = detection_result.face_blendshapes[best_face_idx] if detection_result.face_blendshapes else []
        geom_vec, geom_ok = self._compute_62d_geometry(blendshapes, best_pts_px)

        if partial_face or not geom_ok:
            geom_vec = np.zeros(62, dtype=np.float32)
            geometry_valid = False
        else:
            geometry_valid = True

        return {
            "face_detected": True,
            "fallback_used": False,
            "partial_face": partial_face,
            "image_tensor": image_tensor,
            "aligned_image_rgb": aligned_crop_rgb,
            "geometry_vector": geom_vec,
            "geometry_valid": geometry_valid,
            "head_pose": {"pitch": pitch, "yaw": yaw, "roll": roll},
            "detection_confidence": detection_confidence,
            "bbox": best_bbox,
            "audit_meta": audit_meta,
        }

    def _emergency_fallback(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Direct fallback when landmarker model cannot be initialized."""
        h, w, _ = image_rgb.shape
        side = min(h, w)
        cx, cy = w // 2, h // 2
        crop = image_rgb[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]
        resized = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        norm_img = resized.astype(np.float32) / 255.0
        norm_img = (norm_img - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = np.transpose(norm_img, (2, 0, 1)).astype(np.float32)

        return {
            "face_detected": False,
            "fallback_used": True,
            "partial_face": False,
            "image_tensor": image_tensor,
            "aligned_image_rgb": resized,
            "geometry_vector": np.zeros(62, dtype=np.float32),
            "geometry_valid": False,
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "detection_confidence": None,
            "bbox": (cx - side // 2, cy - side // 2, side, side),
            "audit_meta": {"fallback": True},
        }
