"""
Utility functions and modules for UNet

Implements:
- Downsampling (strided convolution)
- Upsampling (nearest neighbor + convolution)
- Normalization helpers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    """
    Downsample by factor of 2 using strided convolution

    Reduces temporal dimension: [B, C, T] -> [B, C, T//2]
    """
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            return self.conv(x)
        else:
            return self.pool(x)


class Upsample(nn.Module):
    """
    Upsample by factor of 2 using nearest neighbor + convolution

    Increases temporal dimension: [B, C, T] -> [B, C, T*2]
    """
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class GroupNorm32(nn.Module):
    """GroupNorm with 32 groups (or fewer if channels < 32)."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module for identity-init residual branches."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def normalization(channels: int, num_groups: int = 32) -> nn.Module:
    """Create a GroupNorm layer."""
    return nn.GroupNorm(min(num_groups, channels), channels)


def interpolate_frames(x: torch.Tensor, target_length: int, mode: str = 'linear') -> torch.Tensor:
    """Interpolate temporal dimension to target length. [B, C, T] -> [B, C, target_length]."""
    return F.interpolate(
        x, size=target_length, mode=mode,
        align_corners=False if mode == 'linear' else None
    )
