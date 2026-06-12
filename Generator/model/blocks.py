"""
Building Blocks for UNet

Implements:
- ResBlock: Residual block with time and timbre conditioning
- EncoderBlock: Stacks ResBlocks with optional attention and downsampling
- DecoderBlock: Stacks ResBlocks with optional attention and upsampling
"""

import torch
import torch.nn as nn
from .attention import AttentionBlock
from .adaln import AdaLNZero


class ResBlock(nn.Module):
    """
    Residual block with time and timbre conditioning

    Architecture:
        x -> GroupNorm -> SiLU -> Conv1d -> [+ time_emb] ->
             GroupNorm -> SiLU -> Conv1d -> + -> out
        |_______________________________________________|
                         (skip connection)

    With AdaLN conditioning from timbre (optional).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = 256,
        dropout: float = 0.1,
        use_adaln: bool = True,
        conditioning_dim: int = 512
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_adaln = use_adaln

        # First norm and conv
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # Second norm and conv
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

        # AdaLN conditioning (optional)
        if use_adaln:
            self.adaln = AdaLNZero(
                normalized_shape=out_channels,
                conditioning_dim=conditioning_dim,
                use_gate=True
            )
        else:
            self.adaln = None

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        conditioning: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        # Save for skip connection
        skip = x

        # First block: norm -> silu -> conv
        h = self.norm1(x)
        h = torch.nn.functional.silu(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb_out = self.time_emb_proj(time_emb)
        # Broadcast: [B, C] -> [B, C, 1]
        h = h + time_emb_out.unsqueeze(-1)

        # Second block: norm -> silu -> dropout -> conv
        h = self.norm2(h)
        h = torch.nn.functional.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Apply AdaLN if available
        if self.use_adaln and conditioning is not None:
            # AdaLN operates on [B, T, C], so transpose
            h = h.transpose(1, 2)  # [B, T, C]
            h, gate = self.adaln(h, conditioning, return_gate=True)
            h = h.transpose(1, 2)  # [B, C, T]

            # Apply gate to residual
            gate = gate.unsqueeze(-1)  # [B, C, 1]
            h = h * gate

        # Skip connection
        skip = self.skip_connection(skip)

        return skip + h


class EncoderBlock(nn.Module):
    """
    Encoder block for UNet downsampling path

    Consists of:
    - Multiple ResBlocks
    - Optional attention block
    - Optional downsampling
"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = 256,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        use_attention: bool = False,
        attention_heads: int = 8,
        context_dim: int = None,
        downsample: bool = True,
        use_adaln: bool = True,
        conditioning_dim: int = 512
    ):
        super().__init__()
        self.use_attention = use_attention
        self.downsample = downsample

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        channels = in_channels
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(
                    in_channels=channels,
                    out_channels=out_channels,
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                    use_adaln=use_adaln,
                    conditioning_dim=conditioning_dim
                )
            )
            channels = out_channels

        # Attention block (optional)
        if use_attention:
            self.attention = AttentionBlock(
                channels=out_channels,
                context_dim=context_dim,
                num_heads=attention_heads,
                dropout=dropout,
                use_cross_attention=True
            )
        else:
            self.attention = None

        # Downsampling (optional)
        if downsample:
            from .utils import Downsample
            self.downsample_op = Downsample(out_channels, use_conv=True)
        else:
            self.downsample_op = None

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        conditioning: torch.Tensor = None,
        context: torch.Tensor = None,
        **kwargs
    ) -> tuple:
        skips = []

        # Residual blocks
        h = x
        for res_block in self.res_blocks:
            h = res_block(h, time_emb, conditioning)
            skips.append(h)

        # Attention
        if self.attention is not None:
            h = self.attention(h, context=context)

        # Downsampling
        if self.downsample_op is not None:
            h = self.downsample_op(h)

        return h, skips


class DecoderBlock(nn.Module):
    """
    Decoder block for UNet upsampling path

    Consists of:
    - Multiple ResBlocks with skip connections
    - Optional attention block
    - Optional upsampling
    """
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_emb_dim: int = 256,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        use_attention: bool = False,
        attention_heads: int = 8,
        context_dim: int = None,
        upsample: bool = True,
        use_adaln: bool = True,
        conditioning_dim: int = 512
    ):
        super().__init__()
        self.use_attention = use_attention
        self.upsample = upsample

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        channels = in_channels + skip_channels
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(
                    in_channels=channels,
                    out_channels=out_channels,
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                    use_adaln=use_adaln,
                    conditioning_dim=conditioning_dim
                )
            )
            channels = out_channels

        # Attention block (optional)
        if use_attention:
            self.attention = AttentionBlock(
                channels=out_channels,
                context_dim=context_dim,
                num_heads=attention_heads,
                dropout=dropout,
                use_cross_attention=True
            )
        else:
            self.attention = None

        # Upsampling (optional)
        if upsample:
            from .utils import Upsample
            self.upsample_op = Upsample(out_channels, use_conv=True)
        else:
            self.upsample_op = None

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
        conditioning: torch.Tensor = None,
        context: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        # Align temporal length: after upsample h may be 1 frame longer than skip
        # (e.g. stride-2 downsample: 401->201, nearest upsample: 201->402 != 401)
        if x.shape[-1] != skip.shape[-1]:
            x = x[..., :skip.shape[-1]]

        # Concatenate skip connection
        h = torch.cat([x, skip], dim=1)

        # Residual blocks
        for res_block in self.res_blocks:
            h = res_block(h, time_emb, conditioning)

        # Attention
        if self.attention is not None:
            h = self.attention(h, context=context)

        # Upsampling
        if self.upsample_op is not None:
            h = self.upsample_op(h)

        return h
