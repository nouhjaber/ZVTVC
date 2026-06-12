import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import logging

logger = logging.getLogger(__name__)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_grl = ctx.lambda_grl
        # Reverse gradient and scale by lambda_grl
        grad_input = grad_output.neg() * lambda_grl
        return grad_input, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_grl: float = 1.0):
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_grl)

    def set_lambda(self, lambda_grl: float):
        self.lambda_grl = lambda_grl


class SpeakerAdversarial(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_speakers: int = 100,
        dropout: float = 0.3,
        lambda_grl: float = 1.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_speakers = num_speakers

        # Gradient reversal layer
        self.grl = GradientReversalLayer(lambda_grl)

        # Classifier layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_speakers)

        logger.info(f"[SpeakerAdversarial] Initialized: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"num_speakers={num_speakers}, dropout={dropout}, lambda_grl={lambda_grl}")

    def forward(self, z_c: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, 512, T]
        logger.debug(f"[SpeakerAdversarial] Forward - Input: {z_c.shape}")

        # Global average pooling over time dimension
        x = torch.mean(z_c, dim=2)  # [B, 512]
        logger.debug(f"[SpeakerAdversarial] After pooling: {x.shape}")

        # Apply gradient reversal
        x = self.grl(x)  # [B, 512]
        logger.debug(f"[SpeakerAdversarial] After GRL (lambda={self.grl.lambda_grl:.3f}): {x.shape}")

        # First layer
        x = self.fc1(x)  # [B, 256]
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Second layer (output)
        x = self.fc2(x)  # [B, num_speakers]
        logger.debug(f"[SpeakerAdversarial] Output logits: {x.shape}")

        return x

    def set_lambda_grl(self, lambda_grl: float):
        old_lambda = self.grl.lambda_grl
        self.grl.set_lambda(lambda_grl)
        logger.info(f"[SpeakerAdversarial] lambda_grl changed: {old_lambda:.3f} -> {lambda_grl:.3f}")

    def get_lambda_grl(self) -> float:
        return self.grl.lambda_grl
