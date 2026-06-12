"""
Time and Conditioning Embeddings for Flow Matching Generator

Implements:
- Sinusoidal time embeddings for diffusion timestep t
- MLP projection for time embeddings
- Timbre projection for AdaLN conditioning
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings for timestep t ∈ [0, 1]

    Similar to transformer positional encodings but for continuous time.
    Maps scalar timestep to high-dimensional embedding.
    """
    def __init__(self, embedding_dim: int = 256, max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

        # Precompute frequency bands
        half_dim = embedding_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Ensure t is [B, 1]
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        # Compute arguments: t * freqs -> [B, half_dim]
        args = t * self.freqs.unsqueeze(0)

        # Concatenate sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


class TimeEmbeddingMLP(nn.Module):
    """
    MLP to project time embeddings to desired dimension

    Sinusoidal -> Linear -> SiLU -> Linear
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(t_emb)


class TimbreProjection(nn.Module):
    """
    Projects timbre embedding for AdaLN conditioning

    Timbre [B, 256] -> [B, output_dim]
    Can optionally combine with time embedding.
    """
    def __init__(
        self,
        timbre_dim: int = 256,
        output_dim: int = 512,
        combine_with_time: bool = True,
        time_dim: int = 256
    ):
        super().__init__()
        self.combine_with_time = combine_with_time

        if combine_with_time:
            # Input is timbre + time concatenated
            input_dim = timbre_dim + time_dim
        else:
            input_dim = timbre_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(
        self,
        timbre: torch.Tensor,
        time_emb: torch.Tensor = None
    ) -> torch.Tensor:
        if self.combine_with_time:
            if time_emb is None:
                raise ValueError("time_emb required when combine_with_time=True")
            x = torch.cat([timbre, time_emb], dim=-1)
        else:
            x = timbre

        return self.projection(x)


class TimeEmbedding(nn.Module):
    """
    Complete time embedding module: Sinusoidal + MLP

    t (scalar) -> Sinusoidal -> MLP -> embedding
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        max_period: int = 10000
    ):
        super().__init__()

        self.sinusoidal = SinusoidalTimeEmbedding(embedding_dim, max_period)
        self.mlp = TimeEmbeddingMLP(embedding_dim, hidden_dim, embedding_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.sinusoidal(t)
        t_emb = self.mlp(t_emb)
        return t_emb
