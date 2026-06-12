import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class PhonemeClassifier(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 78, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # First linear layer with LeakyReLU and dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)

        # Second linear layer (output)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        logger.info(f"[PhonemeClassifier] Initialized: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"num_classes={num_classes}, dropout={dropout}")

    def forward(self, z_c: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, 512, T]
        logger.debug(f"[PhonemeClassifier] Forward - Input: {z_c.shape}")

        # Transpose to [B, T, 512] for linear layers
        x = z_c.transpose(1, 2)  # [B, T, 512]

        # First layer
        x = self.fc1(x)  # [B, T, 256]
        x = self.leaky_relu(x)
        x = self.dropout(x)
        logger.debug(f"[PhonemeClassifier] After fc1+dropout: {x.shape}")

        # Second layer (output)
        x = self.fc2(x)  # [B, T, num_classes]

        # Transpose back to [B, num_classes, T]
        x = x.transpose(1, 2)  # [B, num_classes, T]
        logger.debug(f"[PhonemeClassifier] Output logits: {x.shape}")

        return x
