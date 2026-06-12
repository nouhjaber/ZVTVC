"""
Attention Mechanisms for UNet

Implements:
- Multi-head self-attention (for mel-spectrogram features)
- Multi-head cross-attention (for conditioning on content+prosody)
- Attention block combining both with residual connections
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism

    Supports both self-attention and cross-attention.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        B, T_q, C = query.shape

        # For self-attention
        if key is None:
            key = query
        if value is None:
            value = query

        T_k = key.shape[1]

        # Project to Q, K, V
        Q = self.q_proj(query)  # [B, T_q, C]
        K = self.k_proj(key)    # [B, T_k, C]
        V = self.v_proj(value)  # [B, T_k, C]

        # Reshape for multi-head: [B, T, C] -> [B, num_heads, T, head_dim]
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, num_heads, T_q, T_k]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attention_mask is not None:
            # Ensure mask has correct shape [B, 1, T_q, T_k] or [B, num_heads, T_q, T_k]
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: [B, num_heads, T_q, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back: [B, num_heads, T_q, head_dim] -> [B, T_q, C]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, C)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class SelfAttention(nn.Module):
    """
    Self-attention block with pre-norm and residual connection

    x -> LayerNorm -> MultiHeadAttention -> + -> output
    |___________________________________|
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attention = MultiHeadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape

        # Pre-norm
        h = self.norm(x)

        # Rearrange for attention: [B, C, T] -> [B, T, C]
        h = h.transpose(1, 2)

        # Self-attention
        h = self.attention(h)

        # Rearrange back: [B, T, C] -> [B, C, T]
        h = h.transpose(1, 2)

        # Residual
        return x + h


class CrossAttention(nn.Module):
    """
    Cross-attention block for conditioning

    Attends to content+prosody conditioning while processing mel features.

    x (mel) -> LayerNorm -> CrossAttention(Q=x, K=cond, V=cond) -> + -> output
    |_______________________________________________________|
    """
    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()

        self.norm = nn.GroupNorm(min(32, channels), channels)

        # Cross-attention (Q from x, K/V from context)
        self.attention = MultiHeadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout
        )

        # Project context to match channels
        if context_dim != channels:
            self.context_proj = nn.Linear(context_dim, channels)
        else:
            self.context_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, T_x]
            context: Conditioning tensor [B, context_dim, T_ctx]
            context_mask: Optional mask [B, T_x, T_ctx]

        Returns:
            Output tensor [B, C, T_x]
        """
        B, C, T_x = x.shape

        # Pre-norm
        h = self.norm(x)

        # Rearrange: [B, C, T] -> [B, T, C]
        h = h.transpose(1, 2)  # [B, T_x, C]

        # Project and rearrange context: [B, context_dim, T_ctx] -> [B, T_ctx, C]
        context = context.transpose(1, 2)  # [B, T_ctx, context_dim]
        context = self.context_proj(context)  # [B, T_ctx, C]

        # Cross-attention: Q from h, K/V from context
        h = self.attention(
            query=h,
            key=context,
            value=context,
            attention_mask=context_mask
        )

        # Rearrange back: [B, T_x, C] -> [B, C, T_x]
        h = h.transpose(1, 2)

        # Residual
        return x + h


class AttentionBlock(nn.Module):
    """
    Combined self-attention and cross-attention block

    Used in UNet at specified resolutions for global context.

    x -> SelfAttention -> CrossAttention(context) -> output
    """
    def __init__(
        self,
        channels: int,
        context_dim: int = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_cross_attention: bool = True
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        # Self-attention
        self.self_attn = SelfAttention(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout
        )

        # Cross-attention (optional)
        if use_cross_attention and context_dim is not None:
            self.cross_attn = CrossAttention(
                channels=channels,
                context_dim=context_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.cross_attn = None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        context_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Self-attention
        x = self.self_attn(x)

        # Cross-attention
        if self.cross_attn is not None and context is not None:
            x = self.cross_attn(x, context, context_mask)

        return x
