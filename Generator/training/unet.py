"""
UNet Architecture for Flow Matching Generator

Main model that combines all components:
- Time embedding
- Timbre conditioning via AdaLN
- Content+Prosody conditioning via cross-attention
- UNet encoder-decoder architecture
- Predicts velocity field v_t for flow matching

Input:
    - Noisy mel-spectrogram x_t: [B, 80, T]
    - Time t: [B]
    - Content features: [B, 512, T]
    - Prosody features: [B, 32, T]
    - Timbre features: [B, 256]

Output:
    - Predicted velocity v_t: [B, 80, T]
"""

import torch
import torch.nn as nn
from typing import Optional, List

from .embeddings import TimeEmbedding, TimbreProjection
from .blocks import EncoderBlock, DecoderBlock, ResBlock
from .attention import AttentionBlock
from .utils import Downsample, Upsample


class FlowMatchingUNet(nn.Module):
    """
    UNet with Conditional Flow Matching for mel-spectrogram generation

    Architecture:
        Encoder path: 3 blocks with downsampling [T, T/2, T/4]
        Bottleneck: ResBlocks with attention
        Decoder path: 3 blocks with upsampling [T/4, T/2, T]

    """
    def __init__(
        self,
        mel_channels: int = 80,
        model_channels: int = 256,
        num_res_blocks: int = 2,
        channel_mult: List[int] = [1, 2, 2],
        attention_resolutions: List[int] = [4],
        num_heads: int = 8,
        dropout: float = 0.1,
        content_dim: int = 512,
        prosody_dim: int = 32,
        timbre_dim: int = 256,
        time_emb_dim: int = 256,
        use_adaln: bool = True
    ):
        super().__init__()

        self.mel_channels = mel_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.num_levels = len(channel_mult)
        self.use_adaln = use_adaln

        # Context dimension: content + prosody concatenated
        self.context_dim = content_dim + prosody_dim
        self.conditioning_dim = timbre_dim + time_emb_dim  # For AdaLN

        # Time embedding
        self.time_embedding = TimeEmbedding(
            embedding_dim=time_emb_dim,
            hidden_dim=512,
            max_period=10000
        )

        # Timbre projection (combines timbre + time for AdaLN)
        self.timbre_projection = TimbreProjection(
            timbre_dim=timbre_dim,
            output_dim=self.conditioning_dim,
            combine_with_time=True,
            time_dim=time_emb_dim
        )

        # Input projection: mel [B, 80, T] -> [B, model_channels, T]
        self.input_proj = nn.Conv1d(mel_channels, model_channels, kernel_size=3, padding=1)

        # Calculate channel counts for each level
        channels_list = [model_channels * mult for mult in channel_mult]

        # ====================================================================
        # ENCODER (Downsampling path)
        # ====================================================================
        self.encoder_blocks = nn.ModuleList()

        in_ch = model_channels
        for level, out_ch in enumerate(channels_list):
            # Resolution at this level (in terms of downsampling factor)
            resolution = 2 ** level  # 1, 2, 4

            # Use attention at specified resolutions
            use_attention = resolution in attention_resolutions

            # Downsample except at last level (bottleneck)
            downsample = (level < self.num_levels - 1)

            encoder_block = EncoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                time_emb_dim=time_emb_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                use_attention=use_attention,
                attention_heads=num_heads,
                context_dim=self.context_dim,
                downsample=downsample,
                use_adaln=use_adaln,
                conditioning_dim=self.conditioning_dim
            )
            self.encoder_blocks.append(encoder_block)

            in_ch = out_ch

        # ====================================================================
        # BOTTLENECK
        # ====================================================================
        bottleneck_channels = channels_list[-1]
        self.bottleneck = nn.ModuleList([
            ResBlock(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
                use_adaln=use_adaln,
                conditioning_dim=self.conditioning_dim
            ),
            AttentionBlock(
                channels=bottleneck_channels,
                context_dim=self.context_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_cross_attention=True
            ),
            ResBlock(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
                use_adaln=use_adaln,
                conditioning_dim=self.conditioning_dim
            )
        ])

        # ====================================================================
        # DECODER (Upsampling path)
        # ====================================================================
        self.decoder_blocks = nn.ModuleList()

        # Reverse channel list for decoder
        channels_list_reversed = list(reversed(channels_list))

        for level in range(self.num_levels):
            # Resolution at this level
            resolution = 2 ** (self.num_levels - level - 1)  # 4, 2, 1

            # Use attention at specified resolutions
            use_attention = resolution in attention_resolutions

            # Upsample except at last level
            upsample = (level < self.num_levels - 1)

            # Channels
            in_ch = channels_list_reversed[level]
            if level < self.num_levels - 1:
                out_ch = channels_list_reversed[level + 1]
            else:
                out_ch = model_channels

            # Skip channels from encoder (same level)
            skip_ch = channels_list_reversed[level]

            decoder_block = DecoderBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                time_emb_dim=time_emb_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                use_attention=use_attention,
                attention_heads=num_heads,
                context_dim=self.context_dim,
                upsample=upsample,
                use_adaln=use_adaln,
                conditioning_dim=self.conditioning_dim
            )
            self.decoder_blocks.append(decoder_block)

        # ====================================================================
        # OUTPUT
        # ====================================================================
        self.output_norm = nn.GroupNorm(min(32, model_channels), model_channels)
        self.output_proj = nn.Conv1d(model_channels, mel_channels, kernel_size=3, padding=1)

        # Zero init output projection for stable training
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor
    ) -> torch.Tensor:
        B, C, T = x.shape

        # ================================================================
        # Prepare conditioning
        # ================================================================

        # Time embedding
        time_emb = self.time_embedding(t)  # [B, time_emb_dim]

        # Timbre conditioning (for AdaLN): timbre + time
        conditioning = self.timbre_projection(timbre, time_emb)  # [B, conditioning_dim]

        # Context for cross-attention: content + prosody
        context = torch.cat([content, prosody], dim=1)  # [B, content_dim + prosody_dim, T]

        # ================================================================
        # Input projection
        # ================================================================
        h = self.input_proj(x)  # [B, model_channels, T]

        # ================================================================
        # ENCODER
        # ================================================================
        encoder_outputs = []

        for level, encoder_block in enumerate(self.encoder_blocks):
            h, skips = encoder_block(
                h,
                time_emb=time_emb,
                conditioning=conditioning,
                context=context
            )
            # Store the last skip connection from each encoder block
            encoder_outputs.append(skips[-1])

            # Downsample context to match next level's resolution
            # EncoderBlock downsamples h internally, so context must follow
            if encoder_block.downsample_op is not None:
                context = torch.nn.functional.avg_pool1d(context, kernel_size=2)

        # ================================================================
        # BOTTLENECK
        # ================================================================
        for layer in self.bottleneck:
            if isinstance(layer, ResBlock):
                h = layer(h, time_emb=time_emb, conditioning=conditioning)
            elif isinstance(layer, AttentionBlock):
                h = layer(h, context=context)

        # ================================================================
        # DECODER
        # ================================================================
        for level, decoder_block in enumerate(self.decoder_blocks):
            # Get skip connection from corresponding encoder level
            skip = encoder_outputs[-(level + 1)]

            h = decoder_block(
                h,
                skip=skip,
                time_emb=time_emb,
                conditioning=conditioning,
                context=context
            )

            # Upsample context to match next level's resolution
            # DecoderBlock upsamples h internally, so context must follow
            if decoder_block.upsample_op is not None:
                context = torch.nn.functional.interpolate(
                    context, scale_factor=2, mode='nearest'
                )

        # ================================================================
        # OUTPUT
        # ================================================================
        h = self.output_norm(h)
        h = torch.nn.functional.silu(h)
        velocity = self.output_proj(h)  # [B, mel_channels, T']

        # Ensure output T matches input T exactly
        # (strided conv + nearest upsample can add 1 frame for odd T)
        if velocity.shape[-1] != T:
            velocity = velocity[..., :T]

        return velocity


class FlowMatchingGenerator(nn.Module):
    """
    Complete Flow Matching Generator with ODE solver
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.unet = FlowMatchingUNet(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor
    ) -> torch.Tensor:
        return self.unet(x, t, content, prosody, timbre)

    @torch.no_grad()
    def sample(
        self,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor,
        num_steps: int = 10,
        solver: str = 'euler'
    ) -> torch.Tensor:
        B = content.shape[0]
        T = content.shape[2]
        device = content.device

        # Start from noise
        x = torch.randn(B, self.unet.mel_channels, T, device=device)

        # Time steps
        dt = 1.0 / num_steps
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

        # Integrate ODE
        for i in range(num_steps):
            t = timesteps[i]
            t_batch = torch.full((B,), t, device=device)

            if solver == 'euler':
                # Euler method: x_{t+1} = x_t + dt * v(x_t, t)
                v = self.unet(x, t_batch, content, prosody, timbre)
                x = x + dt * v

            elif solver == 'midpoint':
                # Midpoint method (RK2)
                v1 = self.unet(x, t_batch, content, prosody, timbre)
                x_mid = x + (dt / 2) * v1

                t_mid = t + dt / 2
                t_mid_batch = torch.full((B,), t_mid, device=device)
                v2 = self.unet(x_mid, t_mid_batch, content, prosody, timbre)

                x = x + dt * v2

            elif solver == 'heun':
                # Heun's method (RK2)
                v1 = self.unet(x, t_batch, content, prosody, timbre)
                x_next = x + dt * v1

                t_next = t + dt
                t_next_batch = torch.full((B,), t_next, device=device)
                v2 = self.unet(x_next, t_next_batch, content, prosody, timbre)

                x = x + (dt / 2) * (v1 + v2)

            else:
                raise ValueError(f"Unknown solver: {solver}")

        return x