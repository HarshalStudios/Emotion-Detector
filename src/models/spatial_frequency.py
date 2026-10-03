"""Spatial + Frequency Dual-Branch Model Module (Experiment A2).

Implements research branch A2:
- Spatial Branch: MobileNetV3-Large pretrained feature extractor with its final
  classification layer replaced by nn.Identity(), producing a 1280-D spatial embedding.
- Frequency Branch: 2D FFT (rfft2) representation followed by learnable complex
  spectral filtering, log-magnitude dynamic range compression, and a lightweight
  4-stage convolutional spectral encoder yielding a 256-D frequency embedding.
- Fusion: Concatenation of spatial (1280-D) and frequency (256-D) embeddings (1536-D).
- Classification Head: Multi-layer perceptron (Linear -> LayerNorm -> Hardswish -> Dropout -> Linear)
  mapping 1536-D to exactly 7 canonical emotion logits.

Strict Constraints:
- Input shape: (B, 3, 224, 224) float32 normalized image tensor.
- Output logits shape: (B, 7).
- Lightweight parameter footprint (~5.60M parameters) suitable for Tesla T4 GPU training.
- Deterministic initialization for seed 42.
- NO geometry components (geometry is strictly isolated to A3/A4/A5).
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class LearnableSpectralFilter(nn.Module):
    """
    Learnable 2D Spectral Filter operating in the Fourier frequency domain.

    Given input image tensor x of shape (B, C, H, W):
    1. Computes 2D real-input FFT: X = torch.fft.rfft2(x, norm="ortho")
       X has complex shape (B, C, H, W_freq) where W_freq = W // 2 + 1 (113 for W=224).
    2. Applies elementwise complex multiplication with learnable weights W = W_r + i * W_i:
       X_filtered = X * W
    3. Computes the log-magnitude spectral energy representation:
       S = torch.log1p(torch.abs(X_filtered) + 1e-8)
       S has real shape (B, C, H, W_freq).
    """

    def __init__(self, in_channels: int = 3, height: int = 224, width: int = 224):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width_freq = width // 2 + 1  # 113 for width=224

        # Learnable complex spectral weights W = W_r + i * W_i
        # Initialized near identity (W_r ~ 1.0, W_i ~ 0.0) with small Gaussian perturbation
        self.weight_real = nn.Parameter(
            torch.ones(in_channels, self.height, self.width_freq, dtype=torch.float32)
            + 0.01 * torch.randn(in_channels, self.height, self.width_freq, dtype=torch.float32)
        )
        self.weight_imag = nn.Parameter(
            0.01 * torch.randn(in_channels, self.height, self.width_freq, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Real image tensor of shape (B, C, H, W).
        Returns:
            Log-magnitude filtered spectral tensor of shape (B, C, H, W // 2 + 1).
        """
        # 1. 2D real-to-complex FFT over spatial dimensions (H, W)
        X = torch.fft.rfft2(x, norm="ortho")  # Complex tensor: (B, C, H, W_freq)

        # 2. Form complex filter weight tensor
        W = torch.complex(self.weight_real, self.weight_imag)  # (C, H, W_freq)

        # 3. Elementwise spectral filtering (broadcast across batch dimension)
        X_filtered = X * W  # (B, C, H, W_freq)

        # 4. Magnitude spectrum computation
        magnitude = torch.abs(X_filtered)

        # 5. Log-magnitude compression: log(1 + |X_filtered|)
        log_magnitude = torch.log1p(magnitude)

        return log_magnitude


class FrequencyBranch(nn.Module):
    """
    Frequency feature extraction branch.
    Transforms input RGB image (B, 3, 224, 224) via:
    1. LearnableSpectralFilter: produces (B, 3, 224, 113) log-magnitude spectrum.
    2. Lightweight 4-stage convolutional encoder with progressive downsampling.
    3. Global adaptive average pooling and linear projection -> (B, embedding_dim).
    """

    def __init__(
        self,
        in_channels: int = 3,
        height: int = 224,
        width: int = 224,
        embedding_dim: int = 256,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.spectral_filter = LearnableSpectralFilter(
            in_channels=in_channels, height=height, width=width
        )

        # 4-stage convolutional spectral feature encoder
        # Input: (B, 3, 224, 113)
        self.conv_encoder = nn.Sequential(
            # Stage 1: (B, 3, 224, 113) -> (B, 32, 112, 57)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # Stage 2: (B, 32, 112, 57) -> (B, 64, 56, 29)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Stage 3: (B, 64, 56, 29) -> (B, 128, 28, 15)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Stage 4: (B, 128, 28, 15) -> (B, 256, 14, 8)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # Global aggregation: (B, 256, 14, 8) -> (B, 256, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.projection = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input image tensor of shape (B, 3, 224, 224).
        Returns:
            Frequency embedding tensor of shape (B, embedding_dim).
        """
        spec = self.spectral_filter(x)
        features = self.conv_encoder(spec)
        embedding = self.projection(features)
        return embedding


class SpatialFrequencyModel(nn.Module):
    """
    Experiment A2: Spatial + Frequency Dual-Branch Architecture.

    Combines:
    1. Spatial Branch: Pretrained MobileNetV3-Large (1280-D spatial embedding).
    2. Frequency Branch: 2D FFT + Learnable Spectral Filter + ConvNet (256-D frequency embedding).
    3. Fusion Head: Concatenates (1280 + 256 = 1536-D) -> MLP -> 7 canonical logits.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freq_embedding_dim: int = 256,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freq_embedding_dim = freq_embedding_dim
        self.dropout_rate = dropout_rate

        # 1. Spatial Branch: MobileNetV3-Large
        # Classifier structure in torchvision:
        # Sequential(
        #   (0): Linear(in_features=960, out_features=1280)
        #   (1): Hardswish()
        #   (2): Dropout(p=0.2)
        #   (3): Linear(in_features=1280, out_features=1000)
        # )
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_large(weights=weights)

        # Replace final 1000-class linear with nn.Identity() to obtain 1280-D spatial embedding
        spatial_dim = mobilenet.classifier[3].in_features  # 1280
        mobilenet.classifier[3] = nn.Identity()
        self.spatial_branch = mobilenet
        self.spatial_dim = spatial_dim

        # 2. Frequency Branch
        self.frequency_branch = FrequencyBranch(
            in_channels=3,
            height=224,
            width=224,
            embedding_dim=freq_embedding_dim,
            dropout_rate=dropout_rate,
        )

        # 3. Fusion & Classification Head
        fused_dim = self.spatial_dim + self.freq_embedding_dim  # 1280 + 256 = 1536
        self.fused_dim = fused_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input image tensor of shape (B, 3, 224, 224).
        Returns:
            Logits tensor of shape (B, num_classes).
        """
        # Spatial embedding: (B, 1280)
        spatial_emb = self.spatial_branch(x)

        # Frequency embedding: (B, 256)
        freq_emb = self.frequency_branch(x)

        # Concatenation: (B, 1536)
        fused_emb = torch.cat([spatial_emb, freq_emb], dim=1)

        # Output logits: (B, 7)
        logits = self.fusion_head(fused_emb)
        return logits

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extracts intermediate embeddings for representation inspection.
        Args:
            x: Input image tensor of shape (B, 3, 224, 224).
        Returns:
            Dict containing spatial_embedding, freq_embedding, and fused_embedding.
        """
        spatial_emb = self.spatial_branch(x)
        freq_emb = self.frequency_branch(x)
        fused_emb = torch.cat([spatial_emb, freq_emb], dim=1)
        return {
            "spatial_embedding": spatial_emb,
            "frequency_embedding": freq_emb,
            "fused_embedding": fused_emb,
        }

    def get_param_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        freq_lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ) -> List[Dict[str, Any]]:
        """
        Returns parameter groups with differentiated learning rates:
        - spatial_branch: backbone_lr (fine-tuning pretrained weights)
        - frequency_branch: freq_lr (training spectral encoder from scratch)
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
                "params": [p for p in self.fusion_head.parameters() if p.requires_grad],
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "fusion_head",
            },
        ]


def create_spatial_frequency_model(
    num_classes: int = 7,
    pretrained: bool = True,
    freq_embedding_dim: int = 256,
    dropout_rate: float = 0.2,
) -> SpatialFrequencyModel:
    """Factory function for creating SpatialFrequencyModel."""
    return SpatialFrequencyModel(
        num_classes=num_classes,
        pretrained=pretrained,
        freq_embedding_dim=freq_embedding_dim,
        dropout_rate=dropout_rate,
    )
