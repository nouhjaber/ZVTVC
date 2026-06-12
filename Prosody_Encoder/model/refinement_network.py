"""
Refinement Network
Light CNN that processes explicit prosody features with residual connections
Architecture: 3 Conv1D layers with dilated convolutions + reconstruction heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class Conv1DBlock(nn.Module):
    """Single Conv1D block with LeakyReLU activation, LayerNorm, and dropout"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int = 1,
        dropout: float = 0.0,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        # Normalization: always LayerNorm over channels
        self.norm = nn.LayerNorm(out_channels)

        # Activation: always LeakyReLU
        self.activation = nn.LeakyReLU(leaky_relu_slope)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        # Apply LayerNorm (expects [B, T, C])
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, C, T]

        # Activation
        x = self.activation(x)

        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)

        return x


class ReconstructionHead(nn.Module):
    """Reconstruction head for individual features"""

    def __init__(self, in_channels: int, feature_type: str = "continuous"):
        super().__init__()
        self.feature_type = feature_type
        self.projection = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.projection(x)

        # Apply sigmoid for binary features (voicing)
        if self.feature_type == "binary":
            output = torch.sigmoid(output)

        return output


class RefinementNetwork(nn.Module):
    """
    Light CNN refinement network for prosody features
    Input: [B, 4, T] - Explicit features [f0_norm, energy_norm, voicing, rhythm]
    Output: [B, 32, T] - Refined prosody representation
    """

    def __init__(
        self,
        explicit_dim: int = 4,
        refined_dim: int = 32,
        leaky_relu_slope: float = 0.2,
        use_residual: bool = True,
        use_reconstruction_heads: bool = True,
    ):
        """
        Initialize Refinement Network
        """
        super().__init__()

        self.explicit_dim = explicit_dim
        self.refined_dim = refined_dim
        self.use_residual = use_residual
        self.use_reconstruction_heads = use_reconstruction_heads

        # Layer 1: [4, T] -> [32, T]
        self.conv1 = Conv1DBlock(
            in_channels=explicit_dim,
            out_channels=refined_dim,
            kernel_size=3,
            padding=1,
            leaky_relu_slope=leaky_relu_slope,
        )

        # Layer 2: [32, T] -> [32, T] with dilation=1
        self.conv2 = Conv1DBlock(
            in_channels=refined_dim,
            out_channels=refined_dim,
            kernel_size=5,
            padding=2,
            dilation=1,
            dropout=0.1,
            leaky_relu_slope=leaky_relu_slope,
        )

        # Layer 3: [32, T] -> [32, T] with dilation=2
        self.conv3 = Conv1DBlock(
            in_channels=refined_dim,
            out_channels=refined_dim,
            kernel_size=5,
            padding=4,
            dilation=2,
            dropout=0.1,
            leaky_relu_slope=leaky_relu_slope,
        )

        # Residual projection: [4, T] -> [32, T]
        if self.use_residual:
            self.residual_proj = nn.Conv1d(explicit_dim, refined_dim, kernel_size=1)

        # Reconstruction heads (for training loss)
        if self.use_reconstruction_heads:
            self.f0_head = ReconstructionHead(refined_dim, feature_type="continuous")
            self.energy_head = ReconstructionHead(refined_dim, feature_type="continuous")
            self.voicing_head = ReconstructionHead(refined_dim, feature_type="binary")
            self.rhythm_head = ReconstructionHead(refined_dim, feature_type="continuous")

    def forward(
        self,
        explicit_features: torch.Tensor,
        return_reconstructions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        x = explicit_features

        # Save input for residual
        residual = x

        # Pass through conv layers
        x = self.conv1(x)  # [B, 32, T]
        x = self.conv2(x)  # [B, 32, T]
        x = self.conv3(x)  # [B, 32, T]

        # Add residual connection
        if self.use_residual:
            residual_proj = self.residual_proj(residual)
            x = x + residual_proj

        refined = x

        # Compute reconstructions if needed
        reconstructions = None
        if return_reconstructions and self.use_reconstruction_heads:
            reconstructions = {
                "f0": self.f0_head(refined),      # [B, 1, T]
                "energy": self.energy_head(refined),  # [B, 1, T]
                "voicing": self.voicing_head(refined),  # [B, 1, T]
                "rhythm": self.rhythm_head(refined),    # [B, 1, T]
            }

        return refined, reconstructions

    def get_parameter_count(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def test_refinement_network():
    """Test the refinement network"""
    print("Testing Refinement Network...")

    # Create model
    model = RefinementNetwork(
        explicit_dim=4,
        refined_dim=32,
        use_residual=True,
        use_reconstruction_heads=True,
    )

    # Print parameter count
    param_count = model.get_parameter_count()
    print(f"Total parameters: {param_count:,}")

    # Test forward pass
    batch_size = 8
    seq_length = 500
    explicit_features = torch.randn(batch_size, 4, seq_length)

    # Inference mode
    refined, _ = model(explicit_features, return_reconstructions=False)
    print(f"Input shape: {explicit_features.shape}")
    print(f"Output shape: {refined.shape}")

    # Training mode with reconstructions
    refined, reconstructions = model(explicit_features, return_reconstructions=True)
    print(f"\nReconstructions:")
    for key, value in reconstructions.items():
        print(f"  {key}: {value.shape}")

    print("\nTest passed!")


if __name__ == "__main__":
    test_refinement_network()