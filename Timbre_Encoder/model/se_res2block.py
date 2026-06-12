"""
SE-Res2Block
============

Squeeze-Excitation Res2Net Block - the core building block of ECAPA-TDNN.

Components:
    1. Res2Net: Multi-scale convolutions (captures different temporal resolutions)
    2. Squeeze-Excitation: Channel attention mechanism
    3. Residual connection: Skip connection for gradient flow

Architecture:
    Input [B, C_in, T]
        ↓
    Conv1x1 → [B, C_hidden, T]
        ↓
    Res2Net → [B, C_hidden, T]  (multi-scale processing)
        ↓
    Conv1x1 → [B, C_out, T]
        ↓
    SE Attention → [B, C_out, T]  (channel reweighting)
        ↓
    + Residual → [B, C_out, T]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger(__name__)


class TransposedLayerNorm(nn.Module):
    """LayerNorm for Conv1d outputs: works on [B, C, T] by transposing to [B, T, C] and back."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class Res2NetBlock(nn.Module):
    """
    Res2Net multi-scale convolution block.
    
    Splits channels into multiple groups and processes hierarchically.
    Each group sees outputs from previous groups, creating multi-scale features.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 4,
    ):
        super().__init__()
        
        assert channels % scale == 0, "channels must be divisible by scale"
        
        self.scale = scale
        self.channels = channels
        self.width = channels // scale
        
        # Create convolutions for each scale (except first, which is identity)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                self.width,
                self.width,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size - 1) // 2,
                bias=False,
            )
            for _ in range(scale - 1)
        ])
        
        self.bn = nn.ModuleList([
            TransposedLayerNorm(self.width)
            for _ in range(scale - 1)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into groups
        spx = torch.split(x, self.width, dim=1)
        
        outputs = []
        sp = spx[0]  # First group is identity
        outputs.append(sp)
        
        # Process each group hierarchically
        for i, (conv, bn) in enumerate(zip(self.convs, self.bn)):
            if i == 0:
                sp = spx[i + 1]
            else:
                sp = sp + spx[i + 1]
            
            sp = conv(sp)
            sp = bn(sp)
            sp = F.leaky_relu(sp)
            outputs.append(sp)
        
        # Concatenate all groups
        out = torch.cat(outputs, dim=1)
        
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Learns to reweight channels based on global context.
    
    Architecture:
        Input [B, C, T]
            ↓
        Global Average Pool → [B, C, 1]
            ↓
        FC (C → C/r) → ReLU
            ↓
        FC (C/r → C) → Sigmoid
            ↓
        Multiply with input → [B, C, T]
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 8,
    ):
        super().__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # Squeeze: Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excitation: Two fully connected layers
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
            
        Returns:
            out: [B, C, T]
        """
        batch, channels, time = x.size()
        
        # Squeeze: Global average pooling
        squeeze = self.avg_pool(x)  # [B, C, 1]
        squeeze = squeeze.view(batch, channels)  # [B, C]
        
        # Excitation: Channel attention
        excitation = F.leaky_relu(self.fc1(squeeze))  # [B, C/r]
        excitation = torch.sigmoid(self.fc2(excitation))  # [B, C]
        excitation = excitation.view(batch, channels, 1)  # [B, C, 1]
        
        # Scale input
        out = x * excitation  # [B, C, T]
        
        return out


class SERes2Block(nn.Module):
    """
    SE-Res2Block: The main building block of ECAPA-TDNN.
    
    Combines:
        - Res2Net for multi-scale temporal processing
        - SE block for channel attention
        - Residual connection for gradient flow
    
    Architecture:
        Input [B, C_in, T]
            ↓
        Conv1x1 (C_in → C_hidden) + BN + ReLU
            ↓
        Res2Net (multi-scale convolution)
            ↓
        Conv1x1 (C_hidden → C_out) + BN
            ↓
        SE Block (channel attention)
            ↓
        + Skip Connection → [B, C_out, T]
            ↓
        ReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 4,
        se_reduction: int = 8,
    ):
        super().__init__()
        logger.debug(f"Initializing SERes2Block: in={in_channels}, out={out_channels}, dilation={dilation}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Bottleneck dimension (typically same as output)
        hidden_channels = out_channels
        
        # 1x1 conv to project input
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = TransposedLayerNorm(hidden_channels)
        
        # Res2Net block
        self.res2net = Res2NetBlock(
            channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            scale=scale,
        )
        
        # 1x1 conv to project output
        self.conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = TransposedLayerNorm(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels, reduction=se_reduction)
        
        # Skip connection (if dimensions don't match, use 1x1 conv)
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                TransposedLayerNorm(out_channels),
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input for skip connection
        identity = x
        
        # 1x1 conv + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out)
        
        # Res2Net multi-scale convolution
        out = self.res2net(out)
        
        # 1x1 conv + BN
        out = self.conv2(out)
        out = self.bn2(out)
        
        # SE attention
        out = self.se(out)
        
        # Skip connection
        identity = self.skip(identity)
        out = out + identity
        
        # Final activation
        out = F.leaky_relu(out)
        
        return out