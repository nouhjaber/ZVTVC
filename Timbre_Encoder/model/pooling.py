"""
Attentive Statistics Pooling
=============================

Aggregates variable-length frame-level features into fixed-size utterance-level embeddings.

Key idea: Learn to weight informative frames more heavily.

Architecture:
    Input: [B, C, T] (frame-level features)
        ↓
    Attention weights: [B, T] (learned importance of each frame)
        ↓
    Weighted mean: [B, C]
    Weighted std: [B, C]
        ↓
    Concatenate: [B, 2C]
    Output: [B, 2C] (utterance-level embedding)

Advantages over simple mean pooling:
    1. Learns to focus on informative frames
    2. Handles variable-length inputs naturally
    3. Captures both mean and variance (richer representation)
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


class AttentiveStatisticsPooling(nn.Module):    
    def __init__(
        self,
        channels: int,
        attention_channels: int = 128,
    ):
        super().__init__()
        logger.debug(f"Initializing AttentiveStatisticsPooling: channels={channels}, attention_channels={attention_channels}")
        self.channels = channels
        self.attention_channels = attention_channels
        
        # Attention network
        # Learns to compute attention weight for each frame
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, kernel_size=1),
            nn.LeakyReLU(),
            TransposedLayerNorm(attention_channels),
            nn.Tanh(),
            nn.Conv1d(attention_channels, channels, kernel_size=1),
            nn.Softmax(dim=2),  # Softmax over time dimension
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"Attentive pooling: input shape {x.shape}")
        # Compute attention weights
        # [B, C, T] → [B, C, T]
        w = self.attention(x)
        
        # Compute weighted mean
        # μ = Σ(w_t * x_t) / Σ(w_t)
        # Since softmax: Σ(w_t) = 1, so μ = Σ(w_t * x_t)
        mean = torch.sum(x * w, dim=2)  # [B, C]
        
        # Compute weighted standard deviation
        # σ² = Σ(w_t * (x_t - μ)²) / Σ(w_t)
        # σ² = Σ(w_t * x_t²) - μ²
        variance = torch.sum(((x - mean.unsqueeze(2)) ** 2) * w, dim=2)  # [B, C]
        std = torch.sqrt(variance + 1e-6)  # [B, C]
        
        # Concatenate mean and std
        stats = torch.cat([mean, std], dim=1)  # [B, 2C]
        
        return stats


class TemporalAveragePooling(nn.Module):
    """
    Simple temporal average pooling (baseline).
    
    Computes mean and standard deviation over time dimension.
    No learned attention weights.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean
        mean = x.mean(dim=2)  # [B, C]
        
        # Compute std
        std = x.std(dim=2)  # [B, C]
        
        # Concatenate
        stats = torch.cat([mean, std], dim=1)  # [B, 2C]
        
        return stats


class SelfAttentivePooling(nn.Module):
    """
    Self-attentive pooling (alternative to attentive stats pooling).
    
    Uses self-attention to compute attention weights.
    More expressive but also more parameters.
    """
    
    def __init__(
        self,
        channels: int,
        attention_heads: int = 4,
    ):
        super().__init__()
        
        self.channels = channels
        self.attention_heads = attention_heads
        
        assert channels % attention_heads == 0
        self.head_dim = channels // attention_heads
        
        # Query, Key, Value projections
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        
        # Output projection
        self.out = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time = x.size()
        
        # Transpose to [B, T, C] for attention
        x = x.transpose(1, 2)  # [B, T, C]
        
        # Compute Q, K, V
        q = self.query(x)  # [B, T, C]
        k = self.key(x)    # [B, T, C]
        v = self.value(x)  # [B, T, C]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, time, self.attention_heads, self.head_dim)
        k = k.view(batch_size, time, self.attention_heads, self.head_dim)
        v = v.view(batch_size, time, self.attention_heads, self.head_dim)
        
        # Transpose to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)  # [B, H, T, T]
        
        # Apply attention to values
        attended = torch.matmul(attention, v)  # [B, H, T, D]
        
        # Reshape back to [B, T, C]
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, time, channels)
        
        # Output projection
        out = self.out(attended)  # [B, T, C]
        
        # Pool over time
        mean = out.mean(dim=1)  # [B, C]
        std = out.std(dim=1)    # [B, C]
        
        stats = torch.cat([mean, std], dim=1)  # [B, 2C]
        
        return stats


class AdaptivePooling(nn.Module):
    def __init__(
        self,
        channels: int,
        pooling_types: list = ['attentive', 'average'],
    ):
        super().__init__()
        
        self.channels = channels
        self.pooling_types = pooling_types
        
        # Create pooling modules
        self.poolings = nn.ModuleDict()
        
        if 'attentive' in pooling_types:
            self.poolings['attentive'] = AttentiveStatisticsPooling(channels)
        
        if 'average' in pooling_types:
            self.poolings['average'] = TemporalAveragePooling()
        
        if 'self_attentive' in pooling_types:
            self.poolings['self_attentive'] = SelfAttentivePooling(channels)
        
        # Output dimension
        self.output_dim = 2 * channels * len(pooling_types)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        
        for name, pooling in self.poolings.items():
            stats = pooling(x)
            outputs.append(stats)
        
        # Concatenate all pooling outputs
        out = torch.cat(outputs, dim=1)
        
        return out