"""Baseline Training Engine Module.

Implements the minimum training and validation loop for the A0 spatial baseline:
- Single-epoch training step with gradient updates and mixed-precision support.
- Single-epoch validation step with full evaluation metrics.
- Cross-entropy loss with label smoothing and class-weighting support.
- Checkpoint serialization and restoration storing:
  - model_state_dict
  - optimizer_state_dict
  - scheduler_state_dict
  - epoch
  - validation_metrics
  - config/metadata
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_classification_metrics


def compute_balanced_class_weights(class_counts: List[int]) -> torch.Tensor:
    """
    Computes inverse class frequency weights following the scikit-learn standard:
    weight[c] = total_samples / (num_classes * count[c])
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    total_samples = np.sum(counts)
    num_classes = len(counts)
    weights = total_samples / (num_classes * counts)
    # Normalize weights so their mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def create_criterion(
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
) -> nn.CrossEntropyLoss:
    """
    Creates CrossEntropyLoss with optional label smoothing and class weights.
    """
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


class BaselineTrainer:
    """
    Minimal training engine for spatial baseline experiments.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.scheduler = scheduler
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.config = config or {}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Executes one complete training epoch.
        Returns:
            Dictionary with training loss, accuracy, and macro F1.
        """
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            targets = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item()) * images.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy().tolist())

        if self.scheduler is not None:
            self.scheduler.step()

        dataset_size = len(dataloader.dataset)
        avg_loss = total_loss / max(1, dataset_size)
        metrics = compute_classification_metrics(all_targets, all_preds)

        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }

    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Executes one complete validation/test evaluation pass.
        Returns:
            Dictionary with validation loss, accuracy, macro F1, weighted F1,
            per-class F1, and confusion matrix.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                targets = batch["label"].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                total_loss += float(loss.item()) * images.size(0)
                preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy().tolist())

        dataset_size = len(dataloader.dataset)
        avg_loss = total_loss / max(1, dataset_size)
        metrics = compute_classification_metrics(all_targets, all_preds)

        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "per_class_f1": metrics["per_class_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
        }

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        val_metrics: Dict[str, Any],
    ) -> None:
        """
        Saves model checkpoint containing model state, optimizer state, scheduler state,
        epoch, validation metrics, and configuration.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "val_metrics": val_metrics,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Loads checkpoint and restores model, optimizer, and scheduler states.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint
