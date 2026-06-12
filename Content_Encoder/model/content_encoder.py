import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from model.encoder import CausalConv1d
from model.multi_scale_backbone import MultiScaleEncoder
from model.fusion import HierarchicalFusion
from model.bottleneck import InformationBottleneck

logger = logging.getLogger(__name__)


class PreProcessing(nn.Module):
    """
    Pre-processing stage: 80 → 256 channels
    - Causal Conv1d with kernel 3, left-only padding
    - LeakyReLU(0.2) + LayerNorm
    """
    def __init__(self, in_channels: int = 80, out_channels: int = 256, kernel_size: int = 3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = CausalConv1d(in_channels, out_channels, kernel_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.layer_norm = nn.LayerNorm(out_channels)

        logger.info(f"[PreProcessing] Initialized: {in_channels} → {out_channels} channels")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, 80, T] - Mel spectrogram
        Output: [B, 256, T] - Preprocessed features
        """
        logger.debug(f"[PreProcessing] Forward - Input: {x.shape}")

        x = self.conv(x)  # [B, 256, T]
        x = self.leaky_relu(x)

        # Apply LayerNorm
        x = x.transpose(1, 2)  # [B, T, 256]
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, 256, T]

        logger.debug(f"[PreProcessing] Output: {x.shape}")
        return x


class OutputProjection(nn.Module):
    """
    Output projection: 256 → 512 channels
    - Linear Conv1d (no activation)
    """
    def __init__(self, in_channels: int = 256, out_channels: int = 512):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # Matches output_norm = nn.LayerNorm(512) used in Content_Encoder/train.py forward()
        self.norm = nn.LayerNorm(out_channels)

        logger.info(f"[OutputProjection] Initialized: {in_channels} → {out_channels} channels")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, 256, T]
        Output: [B, 512, T] - Content features Z_c
        """
        logger.debug(f"[OutputProjection] Forward - Input: {x.shape}")
        x = self.conv(x)        # [B, 512, T]
        x = x.transpose(1, 2)  # [B, T, 512]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, 512, T]
        logger.debug(f"[OutputProjection] Output: {x.shape}")
        return x


class ContentEncoder(nn.Module):
    """
    Complete Content Encoder v3.2

    Pipeline:
    1. PreProcessing: 80 → 256 channels (causal conv + norm)
    2. MultiScaleEncoder: 3 parallel paths (fine, medium, coarse)
    3. HierarchicalFusion: Hierarchical weighted sum
    4. InformationBottleneck: Squeeze-expand with residual
    5. OutputProjection: 256 → 512 channels

    Input: [B, 80, T] - Mel spectrogram
    Output: [B, 512, T] - Content features Z_c
    """
    def __init__(self, alpha_bn: float = 0.5):
        super().__init__()

        self.preprocessing = PreProcessing(in_channels=80, out_channels=256)
        self.multi_scale = MultiScaleEncoder(channels=256)
        self.fusion = HierarchicalFusion()
        self.bottleneck = InformationBottleneck(alpha_bn=alpha_bn)
        self.output_projection = OutputProjection(in_channels=256, out_channels=512)

        logger.info("[ContentEncoder] Initialized complete pipeline")
        logger.info("  PreProcessing: 80 → 256")
        logger.info("  MultiScaleEncoder: 256 → (fine, medium, coarse)")
        logger.info("  HierarchicalFusion: 3 paths → fused")
        logger.info("  InformationBottleneck: 256 → 128 → 256")
        logger.info("  OutputProjection: 256 → 512")

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete encoder

        Args:
            mel: [B, 80, T] - Mel spectrogram

        Returns:
            z_c: [B, 512, T] - Content features
        """
        logger.debug(f"[ContentEncoder] Forward - Input mel: {mel.shape}")

        # 1. PreProcessing: 80 → 256
        x = self.preprocessing(mel)  # [B, 256, T]

        # 2. Multi-scale encoding
        fine, medium, coarse = self.multi_scale(x)  # All [B, 256, T]

        # 3. Hierarchical fusion
        fused = self.fusion(fine, medium, coarse)  # [B, 256, T]

        # 4. Information bottleneck
        bottlenecked = self.bottleneck(fused)  # [B, 256, T]

        # 5. Output projection
        z_c = self.output_projection(bottlenecked)  # [B, 512, T]

        logger.debug(f"[ContentEncoder] Output z_c: {z_c.shape}")
        return z_c

    def set_alpha_bn(self, alpha_bn: float):
        self.bottleneck.set_alpha_bn(alpha_bn)

    def get_fusion_weights(self):
        return self.fusion.get_fusion_weights()
