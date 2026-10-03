"""Spatial + Frequency + Geometry Tri-Branch Model with Learned Gated Fusion (Experiment A5).

Implements research branch A5:
- Spatial Branch: MobileNetV3-Large pretrained feature extractor with its final
  classification layer replaced by nn.Identity(), producing a 1280-D spatial embedding.
- Frequency Branch: 2D FFT (rfft2) representation followed by learnable complex
  spectral filtering, log-magnitude dynamic range compression, and a lightweight
  4-stage convolutional spectral encoder yielding a 256-D frequency embedding.
- Geometry Branch: 62-D facial geometry vector (52 FACS blendshapes + 10 normalized distance ratios)
  projected via an MLP (Linear(62, 128) -> LN -> GELU -> Dropout -> Linear(128, 64) -> LN -> GELU)
  to a 64-D geometry embedding, strictly modulated by a boolean geometry_valid mask.
- Learned Gated Fusion: Per-sample dynamic branch gating network (Linear(1600, 128) -> LN -> GELU -> Linear(128, 3))
  producing inspectable softmax attention weights [g_spatial, g_freq, g_geom].
  Gate calculation and softmax are performed entirely in FP32 with masked_fill(~geom_mask, -1e9),
  guaranteeing that invalid geometry receives exactly 0.0 weight, gates are normalized to 1.0,
  and no FP16 overflow can ever occur under mixed-precision CUDA training.
- Classification Head: Multi-layer perceptron (Linear(1600, 512) -> LayerNorm -> Hardswish -> Dropout -> Linear(512, 7))
  producing exactly 7 canonical emotion logits.

Strict Constraints:
- Input: Image (B, 3, 224, 224), Geometry (B, 62), Geometry Valid (B,) or (B, 1) boolean.
- Output logits shape: (B, 7).
- Gates shape: (B, 3) [spatial, frequency, geometry], inspectable during inference.
- NO temporal modeling, EMA, Fourier augmentation, LBP, Canny, or Sobel features.
- Lightweight footprint (~5.86M parameters) suitable for Tesla T4 GPU training.
- Deterministic initialization for seed 42.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models

from src.models.spatial_frequency import FrequencyBranch, LearnableSpectralFilter
from src.models.spatial_frequency_geometry import GeometryBranch


class LearnedGatedFusion(nn.Module):
    """
    Learned Gated Fusion module for tri-modal emotion recognition.

    Inputs:
        spatial_emb: (B, 1280)
        freq_emb: (B, 256)
        geom_emb: (B, 64)
        geometry_valid: (B,) or (B, 1) boolean mask

    Operation:
    1. Concatenates all branch embeddings: z = [spatial, freq, geom] in R^{B x 1600}.
    2. Passes z through a lightweight GateMLP to produce 3 branch logits s in R^{B x 3}.
    3. Performs gate masking and softmax entirely in FP32:
       a. Upcasts gate_logits to FP32.
       b. Applies masked_fill(~geom_mask, -1e9) in FP32.
       c. Runs torch.softmax(..., dim=-1) in FP32.
       d. Explicitly sets invalid geometry gates to exactly 0.0.
       e. Renormalizes gates so every row sums to 1.0.
       f. Casts final gates back to spatial_emb.dtype.
    4. Modulates each branch:
       gated_spatial = g[:, 0:1] * spatial_emb
       gated_freq = g[:, 1:2] * freq_emb
       gated_geom = g[:, 2:3] * geom_emb
    5. Returns fused_embedding = [gated_spatial, gated_freq, gated_geom] in R^{B x 1600}
       and inspectable gates in R^{B x 3}.
    """

    def __init__(
        self,
        spatial_dim: int = 1280,
        freq_dim: int = 256,
        geom_dim: int = 64,
        gate_hidden_dim: int = 128,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.freq_dim = freq_dim
        self.geom_dim = geom_dim
        self.total_dim = spatial_dim + freq_dim + geom_dim  # 1600

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.total_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(gate_hidden_dim, 3),
        )

    def forward(
        self,
        spatial_emb: torch.Tensor,
        freq_emb: torch.Tensor,
        geom_emb: torch.Tensor,
        geometry_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_emb: (B, 1280)
            freq_emb: (B, 256)
            geom_emb: (B, 64)
            geometry_valid: (B,) or (B, 1) boolean tensor
        Returns:
            fused_embedding: (B, 1600)
            gates: (B, 3) [spatial_gate, frequency_gate, geometry_gate]
        """
        # 1. Joint representation for gate computation
        z = torch.cat([spatial_emb, freq_emb, geom_emb], dim=1)  # (B, 1600)

        # 2. Compute raw branch gate logits
        gate_logits = self.gate_mlp(z)

        # Perform gate masking and softmax in FP32 for CUDA mixed-precision safety.
        gate_logits_fp32 = gate_logits.float()
        if geometry_valid is not None:
            geom_mask = geometry_valid.view(-1).bool()
            masked_geom_logit = gate_logits_fp32[:, 2].masked_fill(
                ~geom_mask,
                -1e9,
            )
            gate_logits_fp32 = torch.stack(
                [
                    gate_logits_fp32[:, 0],
                    gate_logits_fp32[:, 1],
                    masked_geom_logit,
                ],
                dim=-1,
            )
        gates_fp32 = torch.softmax(
            gate_logits_fp32,
            dim=-1,
        )
        if geometry_valid is not None:
            geom_mask_f = geom_mask.to(
                dtype=gates_fp32.dtype
            ).view(-1, 1)
            g_spatial = gates_fp32[:, 0:1]
            g_freq = gates_fp32[:, 1:2]
            g_geom = gates_fp32[:, 2:3] * geom_mask_f
            gates_fp32 = torch.cat(
                [g_spatial, g_freq, g_geom],
                dim=1,
            )
            gates_fp32 = gates_fp32 / (
                gates_fp32.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
            )
        gates = gates_fp32.to(
            dtype=spatial_emb.dtype
        )

        # Modulate representations by learned gates
        gated_spatial = gates[:, 0:1] * spatial_emb  # (B, 1280)
        gated_freq = gates[:, 1:2] * freq_emb        # (B, 256)
        gated_geom = gates[:, 2:3] * geom_emb        # (B, 64)

        # Concatenate gated representations
        fused_embedding = torch.cat([gated_spatial, gated_freq, gated_geom], dim=1)  # (B, 1600)

        return fused_embedding, gates


class SpatialFrequencyGeometryGatedModel(nn.Module):
    """
    Experiment A5: Spatial + Frequency + Geometry Tri-Branch Architecture with Learned Gated Fusion.

    Combines:
    1. Spatial Branch: Pretrained MobileNetV3-Large (1280-D spatial embedding).
    2. Frequency Branch: 2D FFT + Learnable Spectral Filter + ConvNet (256-D frequency embedding).
    3. Geometry Branch: 62-D FACS + Ratios MLP with validity masking (64-D geometry embedding).
    4. Learned Gated Fusion: Per-sample dynamic branch gating with validity masking (1600-D fused representation, 3-D gates).
    5. Fusion Head: MLP mapping 1600-D -> 512-D -> 7 canonical emotion logits.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freq_embedding_dim: int = 256,
        geom_embedding_dim: int = 64,
        gate_hidden_dim: int = 128,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freq_embedding_dim = freq_embedding_dim
        self.geom_embedding_dim = geom_embedding_dim
        self.gate_hidden_dim = gate_hidden_dim
        self.dropout_rate = dropout_rate

        # 1. Spatial Branch: MobileNetV3-Large
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_large(weights=weights)

        # Replace final 1000-class linear layer with nn.Identity() to yield 1280-D spatial embedding
        spatial_dim = mobilenet.classifier[3].in_features  # 1280
        mobilenet.classifier[3] = nn.Identity()
        self.spatial_branch = mobilenet
        self.spatial_dim = spatial_dim

        # 2. Frequency Branch: Exact parity with A2/A4
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

        # 4. Learned Gated Fusion
        self.gated_fusion = LearnedGatedFusion(
            spatial_dim=self.spatial_dim,
            freq_dim=self.freq_embedding_dim,
            geom_dim=self.geom_embedding_dim,
            gate_hidden_dim=gate_hidden_dim,
            dropout_rate=dropout_rate,
        )

        # 5. Classification Head
        fused_dim = self.spatial_dim + self.freq_embedding_dim + self.geom_embedding_dim  # 1600
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
        return_gates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Input image tensor of shape (B, 3, 224, 224).
            geometry: Geometry vector tensor of shape (B, 62).
            geometry_valid: Optional boolean mask tensor of shape (B,) or (B, 1).
            return_gates: If True, returns (logits, gates) for inspection.
        Returns:
            Logits tensor of shape (B, num_classes) or (logits, gates) if return_gates=True.
        """
        # Spatial embedding: (B, 1280)
        spatial_emb = self.spatial_branch(x)

        # Frequency embedding: (B, 256)
        freq_emb = self.frequency_branch(x)

        # Geometry embedding: (B, 64) with validity masking
        geom_emb = self.geometry_branch(geometry, geometry_valid=geometry_valid)

        # Learned Gated Fusion: (B, 1600) and inspectable gates (B, 3)
        fused_emb, gates = self.gated_fusion(
            spatial_emb, freq_emb, geom_emb, geometry_valid=geometry_valid
        )

        # Output logits: (B, 7)
        logits = self.fusion_head(fused_emb)

        if return_gates:
            return logits, gates
        return logits

    def extract_features(
        self,
        x: torch.Tensor,
        geometry: torch.Tensor,
        geometry_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts intermediate embeddings and gates for research representation analysis.
        Args:
            x: Input image tensor of shape (B, 3, 224, 224).
            geometry: Geometry vector tensor of shape (B, 62).
            geometry_valid: Optional boolean mask tensor of shape (B,) or (B, 1).
        Returns:
            Dict containing raw embeddings, gates, gated embeddings, and fused embedding.
        """
        spatial_emb = self.spatial_branch(x)
        freq_emb = self.frequency_branch(x)
        geom_emb = self.geometry_branch(geometry, geometry_valid=geometry_valid)

        fused_emb, gates = self.gated_fusion(
            spatial_emb, freq_emb, geom_emb, geometry_valid=geometry_valid
        )

        return {
            "spatial_embedding": spatial_emb,
            "frequency_embedding": freq_emb,
            "geometry_embedding": geom_emb,
            "gates": gates,
            "gated_spatial": gates[:, 0:1] * spatial_emb,
            "gated_frequency": gates[:, 1:2] * freq_emb,
            "gated_geometry": gates[:, 2:3] * geom_emb,
            "fused_embedding": fused_emb,
        }

    def get_gates(
        self,
        x: torch.Tensor,
        geometry: torch.Tensor,
        geometry_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience inference method to inspect branch gates for research analysis.
        Args:
            x: Input image tensor of shape (B, 3, 224, 224).
            geometry: Geometry vector tensor of shape (B, 62).
            geometry_valid: Optional boolean mask tensor of shape (B,) or (B, 1).
        Returns:
            Tensor of shape (B, 3) containing [spatial_gate, frequency_gate, geometry_gate].
        """
        spatial_emb = self.spatial_branch(x)
        freq_emb = self.frequency_branch(x)
        geom_emb = self.geometry_branch(geometry, geometry_valid=geometry_valid)
        _, gates = self.gated_fusion(
            spatial_emb, freq_emb, geom_emb, geometry_valid=geometry_valid
        )
        return gates

    def get_param_groups(
        self,
        backbone_lr: float = 1e-4,
        freq_lr: float = 1e-3,
        geom_lr: float = 1e-3,
        gate_lr: float = 1e-3,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ) -> List[Dict[str, Any]]:
        """
        Returns parameter groups with differentiated learning rates:
        - spatial_branch: backbone_lr (fine-tuning pretrained weights)
        - frequency_branch: freq_lr (training spectral encoder from scratch)
        - geometry_branch: geom_lr (training geometry MLP from scratch)
        - gated_fusion: gate_lr (training gate generator network from scratch)
        - fusion_head: head_lr (training classification head from scratch)
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
                "params": [p for p in self.gated_fusion.parameters() if p.requires_grad],
                "lr": gate_lr,
                "weight_decay": weight_decay,
                "name": "gated_fusion",
            },
            {
                "params": [p for p in self.fusion_head.parameters() if p.requires_grad],
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "fusion_head",
            },
        ]


def create_spatial_frequency_geometry_gated_model(
    num_classes: int = 7,
    pretrained: bool = True,
    freq_embedding_dim: int = 256,
    geom_embedding_dim: int = 64,
    gate_hidden_dim: int = 128,
    dropout_rate: float = 0.2,
) -> SpatialFrequencyGeometryGatedModel:
    """Factory function for creating SpatialFrequencyGeometryGatedModel."""
    return SpatialFrequencyGeometryGatedModel(
        num_classes=num_classes,
        pretrained=pretrained,
        freq_embedding_dim=freq_embedding_dim,
        geom_embedding_dim=geom_embedding_dim,
        gate_hidden_dim=gate_hidden_dim,
        dropout_rate=dropout_rate,
    )
