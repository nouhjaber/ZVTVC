"""
ECAPA-TDNN Model
================

Emphasized Channel Attention, Propagation and Aggregation in TDNN.

Architecture:
    Input: Mel spectrogram [B, 80, T]
        ↓
    Conv1 (80 → 128) + ReLU
        ↓
    SE-Res2Block 1 (128 → 256, dilation=2)
        ↓
    SE-Res2Block 2 (256 → 256, dilation=3)
        ↓
    SE-Res2Block 3 (256 → 256, dilation=4)
        ↓
    SE-Res2Block 4 (256 → 256, dilation=5)
        ↓
    Attentive Statistics Pooling (256, T) → (512,)
        ↓
    FC (512 → 256) + BN
        ↓
    L2 Normalize
        ↓
    Output: Speaker embedding [B, 256]

Parameters: ~2.5M
Receptive field: ~600ms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger(__name__)

from .se_res2block import SERes2Block
from .pooling import AttentiveStatisticsPooling


class TransposedLayerNorm(nn.Module):
    """LayerNorm for Conv1d outputs: works on [B, C, T] by transposing to [B, T, C] and back."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ECAPATDNN(nn.Module):

    def __init__(
        self,
        input_dim: int = 80,
        embedding_dim: int = 256,
        input_conv_channels: int = 128,
        blocks_config: Optional[list] = None,
        attention_channels: int = 128,
        use_l2_norm: bool = True,
    ):
        super().__init__()
        logger.info("Initializing ECAPA-TDNN model")
        logger.debug(f"Parameters: input_dim={input_dim}, embedding_dim={embedding_dim}, use_l2_norm={use_l2_norm}")

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.use_l2_norm = use_l2_norm
        
        # Default blocks configuration
        if blocks_config is None:
            blocks_config = [
                {'in_channels': 128, 'out_channels': 256, 'dilation': 2, 'kernel_size': 3, 'scale': 4, 'se_reduction': 8},
                {'in_channels': 256, 'out_channels': 256, 'dilation': 3, 'kernel_size': 3, 'scale': 4, 'se_reduction': 8},
                {'in_channels': 256, 'out_channels': 256, 'dilation': 4, 'kernel_size': 3, 'scale': 4, 'se_reduction': 8},
                {'in_channels': 256, 'out_channels': 256, 'dilation': 5, 'kernel_size': 3, 'scale': 4, 'se_reduction': 8},
            ]
        
        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, input_conv_channels, kernel_size=5, padding=2, bias=False),
            TransposedLayerNorm(input_conv_channels),
            nn.LeakyReLU(),
        )
        
        # SE-Res2Blocks
        logger.debug(f"Creating {len(blocks_config)} SE-Res2Blocks")
        self.blocks = nn.ModuleList()
        for i, config in enumerate(blocks_config):
            logger.debug(f"Block {i}: {config}")
            block = SERes2Block(
                in_channels=config['in_channels'],
                out_channels=config['out_channels'],
                kernel_size=config['kernel_size'],
                dilation=config['dilation'],
                scale=config['scale'],
                se_reduction=config['se_reduction'],
            )
            self.blocks.append(block)
        
        # Get output channels from last block
        last_channels = blocks_config[-1]['out_channels']
        
        # Attentive statistics pooling
        self.pooling = AttentiveStatisticsPooling(
            channels=last_channels,
            attention_channels=attention_channels,
        )
        
        # Pooling outputs: mean + std = 2 * last_channels
        pooling_output_dim = 2 * last_channels
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(pooling_output_dim, embedding_dim, bias=False),
            nn.LayerNorm(embedding_dim),
        )
        logger.info(f"ECAPA-TDNN initialized with {self.get_num_params():,} parameters")
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        # Handle [B, T, 80] input (transpose if needed)
        if x.dim() == 3 and x.size(2) == self.input_dim:
            x = x.transpose(1, 2)  # [B, 80, T]

        # Input convolution
        x = self.input_conv(x)  # [B, 128, T]

        # SE-Res2Blocks
        for block in self.blocks:
            x = block(x)  # [B, 256, T]

        frame_features = x

        # Attentive statistics pooling
        pooled = self.pooling(x)  # [B, 512]

        # Embedding head
        embedding = self.embedding(pooled)  # [B, 256]

        # L2 normalization
        if self.use_l2_norm:
            embedding = F.normalize(embedding, p=2, dim=1)

        if return_intermediate:
            return {
                'embedding': embedding,
                'pooled': pooled,
                'frame_features': frame_features,
            }

        return embedding
    
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure 2D
        if embedding1.dim() == 1:
            embedding1 = embedding1.unsqueeze(0)
        if embedding2.dim() == 1:
            embedding2 = embedding2.unsqueeze(0)
        
        # Cosine similarity
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        
        return similarity
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_receptive_field(self) -> float:
        """
        Estimate receptive field in milliseconds.
        
        Approximation based on dilations.
        """
        # Input conv: kernel=5, dilation=1 → RF=5
        rf = 5
        
        # Each SE-Res2Block adds: kernel * dilation
        for block in self.blocks:
            # Get dilation from first Res2Net conv
            dilation = block.res2net.convs[0].dilation[0]
            kernel = block.res2net.convs[0].kernel_size[0]
            rf += kernel * dilation
        
        # Convert frames to milliseconds (hop_length=320, sr=16000)
        # 1 frame = 20ms
        rf_ms = rf * 20.0
        
        return rf_ms


def create_ecapa_tdnn(config: Optional[Dict] = None) -> ECAPATDNN:
    if config is None:
        config = {}
    
    # Default configuration
    default_config = {
        'input_dim': 80,
        'embedding_dim': 256,
        'input_conv_channels': 128,
        'attention_channels': 128,
        'use_l2_norm': True,
        'blocks_config': [
            {'in_channels': 128, 'out_channels': 256, 'dilation': 2, 'kernel_size': 3, 'scale': 4, 'se_reduction': 8},
            {'in_channels': 256, 'out_channels': 256, 'dilation': 3, 'kernel_size': 3, 'scale': 4, 'se_reduction': 8},
            {'in_channels': 256, 'out_channels': 256, 'dilation': 4, 'kernel_size': 3, 'scale': 4, 'se_reduction': 8},
            {'in_channels': 256, 'out_channels': 256, 'dilation': 5, 'kernel_size': 3, 'scale': 4, 'se_reduction': 8},
        ]
    }
    
    # Merge configs
    default_config.update(config)
    
    model = ECAPATDNN(**default_config)
    
    return model


class ECAPatDNNWithClassifier(nn.Module):    
    def __init__(
        self,
        ecapa_tdnn: ECAPATDNN,
        num_speakers: int,
    ):
        super().__init__()
        
        self.ecapa_tdnn = ecapa_tdnn
        self.num_speakers = num_speakers
        
        # Classification head
        self.classifier = nn.Linear(ecapa_tdnn.embedding_dim, num_speakers)
    
    def forward(self, x: torch.Tensor) -> tuple:
        # Extract embedding
        embedding = self.ecapa_tdnn(x)
        
        # Classify
        logits = self.classifier(embedding)
        
        return embedding, logits