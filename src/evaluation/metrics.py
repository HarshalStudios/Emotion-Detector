"""Evaluation Metrics Module.

Computes classification metrics for 7-class facial emotion recognition:
- Accuracy
- Macro F1
- Weighted F1
- Per-class F1 (dictionary mapping class id/name to F1 score)
- Confusion Matrix (7x7 matrix)
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CANONICAL_CLASS_NAMES: List[str] = [
    "Neutral",
    "Happy",
    "Sad",
    "Surprise",
    "Fear",
    "Disgust",
    "Angry",
]


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes standard multi-class evaluation metrics.
    Args:
        y_true: Ground truth integer class labels (0..6).
        y_pred: Predicted integer class labels (0..6).
        class_names: Optional list of class names (defaults to 7 canonical emotions).
    Returns:
        Dict containing:
            "accuracy": float
            "macro_f1": float
            "weighted_f1": float
            "per_class_f1": Dict[str, float]
            "confusion_matrix": List[List[int]] (7x7)
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    if class_names is None:
        class_names = CANONICAL_CLASS_NAMES

    num_classes = len(class_names)
    labels = list(range(num_classes))

    acc = float(accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, labels=labels, average="weighted", zero_division=0))

    per_class_scores = f1_score(y_t, y_p, labels=labels, average=None, zero_division=0)
    per_class_f1 = {
        class_names[i]: float(per_class_scores[i])
        for i in range(num_classes)
    }

    cm = confusion_matrix(y_t, y_p, labels=labels).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }
