"""Spatial Baseline Model Module.

Implements research baseline A0: strong spatial-only visual classifier.
Candidate backbones:
- ConvNeXt-Tiny (convnext_tiny)
- EfficientNet-B0 (efficientnet_b0)
- MobileNetV3-Large (mobilenet_v3_large) [DEVELOPMENT DEFAULT]

Note: MobileNetV3-Large is the current DEVELOPMENT default because of the constrained CPU environment.
This is NOT the experimentally selected final backbone. Final backbone selection remains determined
by the predefined baseline evaluation.

Strict constraints:
- Input shape: (B, 3, 224, 224)
- Pretrained ImageNet weights
- Classification head replaced for 7 canonical emotion classes
- Output logits shape: (B, 7)
- Pure spatial branch: NO frequency, geometry, fusion, or temporal components.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torchvision.models as models

SUPPORTED_BACKBONES = ["convnext_tiny", "efficientnet_b0", "mobilenet_v3_large"]


class SpatialBaseline(nn.Module):
    """
    Configurable spatial baseline classifier for 7-class facial emotion recognition.
    """

    def __init__(
        self,
        # MobileNetV3-Large is the current DEVELOPMENT default because of the constrained CPU environment. This is NOT the experimentally selected final backbone. Final backbone selection remains determined by the predefined baseline evaluation.
        backbone_name: str = "mobilenet_v3_large",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

        if self.backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: '{backbone_name}'. Supported: {SUPPORTED_BACKBONES}"
            )

        if self.backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
            # ConvNeXt classifier structure:
            # Sequential(
            #   (0): LayerNorm2d((768,), eps=1e-06)
            #   (1): Flatten(start_dim=1, end_dim=-1)
            #   (2): Linear(in_features=768, out_features=1000)
            # )
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
            self.model = model
            self.head_module = model.classifier[2]

        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            # EfficientNet classifier structure:
            # Sequential(
            #   (0): Dropout(p=0.2)
            #   (1): Linear(in_features=1280, out_features=1000)
            # )
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            self.model = model
            self.head_module = model.classifier[1]

        elif self.backbone_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_large(weights=weights)
            # MobileNetV3 classifier structure:
            # Sequential(
            #   (0): Linear(in_features=960, out_features=1280)
            #   (1): Hardswish()
            #   (2): Dropout(p=0.2)
            #   (3): Linear(in_features=1280, out_features=1000)
            # )
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
            self.model = model
            self.head_module = model.classifier[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input image tensor of shape (B, 3, 224, 224).
        Returns:
            Logits tensor of shape (B, 7).
        """
        return self.model(x)

    def get_param_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ) -> List[Dict[str, Any]]:
        """
        Returns parameter groups with differentiated learning rates:
        - backbone parameters: backbone_lr
        - newly initialized classification head parameters: head_lr
        """
        head_params_set = set(self.head_module.parameters())
        backbone_params = [p for p in self.parameters() if p not in head_params_set and p.requires_grad]
        head_params = [p for p in self.parameters() if p in head_params_set and p.requires_grad]

        return [
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "name": "backbone",
            },
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            },
        ]


def create_spatial_baseline(
    # MobileNetV3-Large is the current DEVELOPMENT default because of the constrained CPU environment. This is NOT the experimentally selected final backbone. Final backbone selection remains determined by the predefined baseline evaluation.
    backbone_name: str = "mobilenet_v3_large",
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
) -> SpatialBaseline:
    """Factory function for creating SpatialBaseline models."""
    return SpatialBaseline(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )
