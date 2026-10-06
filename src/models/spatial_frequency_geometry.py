"""Spatial + Frequency + Geometry Tri-Branch Model with Plain Concatenation (Experiment A4).

Implements research branch A4:
- Spatial Branch: MobileNetV3-Large pretrained feature extractor with its final
  classification layer replaced by nn.Identity(), producing a 1280-D spatial embedding.
- Frequency Branch: 2D FFT (rfft2) representation followed by learnable complex
  spectral filtering, log-magnitude dynamic range compression, and a lightweight
  4-stage convolutional spectral encoder yielding a 256-D frequency embedding.
- Geometry Branch: 62-D facial geometry vector (52 FACS blendshapes + 10 normalized distance ratios)
  projected via an MLP (Linear(62, 128) -> LN -> GELU -> Dropout -> Linear(128, 64) -> LN -> GELU)
  to a 64-D geometry embedding, strictly modulated by a boolean geometry_valid mask.
- Plain Concatenation Fusion: Concatenates (1280 + 256 + 64 = 1600-D) embeddings without gating.
- Classification Head: Multi-layer perceptron (Linear(1600, 512) -> LayerNorm -> Hardswish -> Dropout -> Linear(512, 7))
  producing exactly 7 canonical emotion logits.

Strict Constraints:
- Input: Image (B, 3, 224, 224), Geometry (B, 62), Geometry Valid (B,) or (B, 1) boolean.
- Output logits shape: (B, 7).
- Plain concatenation only (NO gated fusion; gated fusion is isolated to A5).
- NO temporal modeling, EMA, Fourier augmentation, LBP, Canny, or Sobel features.
- Lightweight footprint (~5.65M parameters) suitable for Tesla T4 GPU training.
- Deterministic initialization for seed 42.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models

from src.models.spatial_frequency import FrequencyBranch, LearnableSpectralFilter


class GeometryBranch(nn.Module):
    """
    Geometry feature extraction branch.
    Maps 62-D facial geometry representation (52 FACS blendshapes + 10 distance ratios)
    to a 64-D compact geometry embedding with validity masking.

    Structure:
        Linear(62, 128) -> LayerNorm(128) -> GELU() -> Dropout(dropout_rate)
        -> Linear(128, 64) -> LayerNorm(64) -> GELU()

    Validity Masking:
        When geometry_valid is False (e.g. face detection fallback was used),
        the geometry embedding is strictly masked to all zeros:
        geometry_embedding = geometry_embedding * mask
        where mask = geometry_valid.view(-1, 1).float()
    """

    def __init__(
        self,
        in_features: int = 62,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.embedding_dim = embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(
        self,
        geometry: torch.Tensor,
        geometry_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            geometry: Float tensor of shape (B, 62).
            geometry_valid: Boolean or float tensor of shape (B,) or (B, 1).
        Returns:
            Geometry embedding tensor of shape (B, 64). If geometry_valid is False,
            the embedding is strictly zeroed out.
        """
        geom_emb = self.mlp(geometry)

        if geometry_valid is not None:
            # Reshape mask to (B, 1) and cast to float
            mask = geometry_valid.view(-1, 1).to(dtype=geom_emb.dtype)
            geom_emb = geom_emb * mask

        return geom_emb


class SpatialFrequencyGeometryModel(nn.Module):
    """
    Experiment A4: Spatial + Frequency + Geometry Tri-Branch Model with Plain Concatenation.

    Combines:
    1. Spatial Branch: Pretrained MobileNetV3-Large (1280-D spatial embedding).
    2. Frequency Branch: 2D FFT + Learnable Spectral Filter + ConvNet (256-D frequency embedding).
    3. Geometry Branch: 62-D FACS + Ratios MLP with validity masking (64-D geometry embedding).
    4. Plain Concatenation: (1280 + 256 + 64) = 1600-D fused representation.
    5. Fusion Head: MLP mapping 1600-D -> 512-D -> 7 canonical emotion logits.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freq_embedding_dim: int = 256,
        geom_embedding_dim: int = 64,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freq_embedding_dim = freq_embedding_dim
        self.geom_embedding_dim = geom_embedding_dim
        self.dropout_rate = dropout_rate

        # 1. Spatial Branch: MobileNetV3-Large
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_large(weights=weights)

        # Replace final 1000-class linear layer with nn.Identity() to yield 1280-D spatial embedding
        spatial_dim = mobilenet.classifier[3].in_features  # 1280
        mobilenet.classifier[3] = nn.Identity()
        self.spatial_branch = mobilenet
        self.spatial_dim = spatial_dim

        # 2. Frequency Branch: Reused exact A2 implementation
        self.frequency_branch = FrequencyBranch(
            in_channels=3,
            height=224,
            width=224,
            embedding_dim=freq_embedding_dim,
            dropout_rate=dropout_rate,
        )

        # 3. Geometry Branch: 62-D -> 64-D with validity masking
        self.geometry_branch = GeometryBranch(
            in_features=62,
            hidden_dim=128,
            embedding_dim=geom_embedding_dim,
            dropout_rate=dropout_rate,
        )

        # 4. Fusion & Classification Head (Plain Concatenation)
        fused_dim = self.spatial_dim + self.freq_embedding_dim + self.geom_embedding_dim  # 1280 + 256 + 64 = 1600
        self.fused_dim = fused_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        geometry: torch.Tensor,
        geometry_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input image tensor of shape (B, 3, 224, 224).
            geometry: Geometry vector tensor of shape (B, 62).
            geometry_valid: Optional boolean mask tensor of shape (B,) or (B, 1).
        Returns:
            Logits tensor of shape (B, num_classes).
        """
        # Spatial embedding: (B, 1280)
        spatial_emb = self.spatial_branch(x)

        # Frequency embedding: (B, 256)
        freq_emb = self.frequency_branch(x)

        # Geometry embedding: (B, 64) with validity masking
        geom_emb = self.geometry_branch(geometry, geometry_valid=geometry_valid)

        # Plain concatenation: (B, 1600)
        fused_emb = torch.cat([spatial_emb, freq_emb, geom_emb], dim=1)

        # Output logits: (B, 7)
        logits = self.fusion_head(fused_emb)
        return logits

    def extract_features(
        self,
        x: torch.Tensor,
        geometry: torch.Tensor,
        geometry_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts intermediate embeddings for representation audit.
        Args:
            x: Input image tensor of shape (B, 3, 224, 224).
            geometry: Geometry vector tensor of shape (B, 62).
            geometry_valid: Optional boolean mask tensor of shape (B,) or (B, 1).
        Returns:
            Dict containing spatial_embedding, frequency_embedding,
            geometry_embedding, and fused_embedding.
        """
        spatial_emb = self.spatial_branch(x)
        freq_emb = self.frequency_branch(x)
        geom_emb = self.geometry_branch(geometry, geometry_valid=geometry_valid)
        fused_emb = torch.cat([spatial_emb, freq_emb, geom_emb], dim=1)

        return {
            "spatial_embedding": spatial_emb,
            "frequency_embedding": freq_emb,
            "geometry_embedding": geom_emb,
            "fused_embedding": fused_emb,
        }

    def get_param_groups(
        self,
        backbone_lr: float = 1e-4,
        freq_lr: float = 1e-3,
        geom_lr: float = 1e-3,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ) -> List[Dict[str, Any]]:
        """
        Returns parameter groups with differentiated learning rates:
        - spatial_branch: backbone_lr (fine-tuning pretrained weights)
        - frequency_branch: freq_lr (training spectral encoder from scratch)
        - geometry_branch: geom_lr (training geometry MLP from scratch)
        - fusion_head: head_lr (training plain concatenation classification head)
        """
        return [
            {
                "params": [p for p in self.spatial_branch.parameters() if p.requires_grad],
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "name": "spatial_branch",
            },
            {
                "params": [p for p in self.frequency_branch.parameters() if p.requires_grad],
                "lr": freq_lr,
                "weight_decay": weight_decay,
                "name": "frequency_branch",
            },
            {
                "params": [p for p in self.geometry_branch.parameters() if p.requires_grad],
                "lr": geom_lr,
                "weight_decay": weight_decay,
                "name": "geometry_branch",
            },
            {
                "params": [p for p in self.fusion_head.parameters() if p.requires_grad],
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "fusion_head",
            },
        ]


def create_spatial_frequency_geometry_model(
    num_classes: int = 7,
    pretrained: bool = True,
    freq_embedding_dim: int = 256,
    geom_embedding_dim: int = 64,
    dropout_rate: float = 0.2,
) -> SpatialFrequencyGeometryModel:
    """Factory function for creating SpatialFrequencyGeometryModel."""
    return SpatialFrequencyGeometryModel(
        num_classes=num_classes,
        pretrained=pretrained,
        freq_embedding_dim=freq_embedding_dim,
        geom_embedding_dim=geom_embedding_dim,
        dropout_rate=dropout_rate,
    )
