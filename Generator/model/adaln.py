"""
Adaptive Layer Normalization (AdaLN) for Timbre Conditioning

Implements:
- AdaLN: Modulates layer norm with learned scale and shift from timbre
- AdaLN-Zero: Zero-initialized gating for better training stability
- Full AdaLN block with 6 parameters: γ1, β1, γ2, β2, α, gate

Reference: "Scalable Diffusion Models with Transformers" (DiT)
"""

import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization

    Modulates LayerNorm output with learned scale and shift from conditioning.

    Standard LayerNorm:
        y = γ * (x - μ) / σ + β

    AdaLN:
        y = (1 + scale) * (x - μ) / σ + shift
        where scale, shift = f(conditioning)
    """
    def __init__(
        self,
        normalized_shape: int,
        conditioning_dim: int,
        zero_init: bool = True
    ):
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)

        # Linear to predict scale and shift
        self.modulation = nn.Linear(conditioning_dim, 2 * normalized_shape)

        # Zero initialization for stability
        if zero_init:
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, ...., normalized_shape]
            conditioning: Conditioning vector [B, conditioning_dim]

        Returns:
            Modulated output [B, ...., normalized_shape]
        """
        # Normalize
        x_norm = self.norm(x)

        # Get modulation parameters
        modulation = self.modulation(conditioning)

        # For broadcasting: conditioning is [B, 2*C], need to make it [B, 1, ..., 1, 2*C]
        # Then split into scale [B, 1, ..., 1, C] and shift [B, 1, ..., 1, C]
        while modulation.ndim < x.ndim:
            modulation = modulation.unsqueeze(1)

        scale, shift = modulation.chunk(2, dim=-1)

        # Apply modulation: (1 + scale) * x_norm + shift
        return (1 + scale) * x_norm + shift


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero: AdaLN with residual gating

    Extends AdaLN with a learned gate parameter that controls the strength
    of the residual connection. Initialized to zero for stable training.

    y = x + gate * block(AdaLN(x, cond))

    This is the "Zero" variant from DiT paper.
    """
    def __init__(
        self,
        normalized_shape: int,
        conditioning_dim: int,
        use_gate: bool = True
    ):
        super().__init__()
        self.use_gate = use_gate

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)

        # Predict scale, shift, and optionally gate
        num_params = 3 if use_gate else 2
        self.modulation = nn.Linear(conditioning_dim, num_params * normalized_shape)

        # Zero initialization
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        return_gate: bool = False
    ):
        # Normalize
        x_norm = self.norm(x)

        # Get modulation parameters [B, num_params * C]
        modulation = self.modulation(conditioning)

        # Split into scale, shift, gate [B, C] each
        if self.use_gate:
            scale, shift, gate = modulation.chunk(3, dim=-1)
        else:
            scale, shift = modulation.chunk(2, dim=-1)
            gate = None

        # Broadcast to match x dimensions
        # x is [B, T, C], scale/shift/gate are [B, C]
        # We need to make them [B, 1, C] for broadcasting
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
            if gate is not None:
                gate = gate.unsqueeze(1)

        # Apply modulation
        output = (1 + scale) * x_norm + shift

        if return_gate:
            # Remove the broadcasting dimension from gate before returning
            # gate is [B, 1, C], squeeze to [B, C]
            if gate is not None:
                gate = gate.squeeze(1)
            return output, gate
        else:
            return output
