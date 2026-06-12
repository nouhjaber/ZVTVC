import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class InformationBottleneck(nn.Module):
    def __init__(self, alpha_bn: float = 0.5):
        super().__init__()
        self.channels = 256
        self.bottleneck_channels = 128

        # Main path: squeeze-expand
        self.squeeze = nn.Conv1d(self.channels, self.bottleneck_channels, kernel_size=1)
        self.expand = nn.Conv1d(self.bottleneck_channels, self.channels, kernel_size=1)

        # Activation
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        # Residual weight (annealed during training)
        # stage_0: 0.5, stage_1: 0.3, stage_2: 0.1
        self.alpha_bn = alpha_bn

        logger.info(f"[InformationBottleneck] Initialized with channels={self.channels}, "
                   f"bottleneck_channels={self.bottleneck_channels}, alpha_bn={alpha_bn}")

    def forward(self, x):
        logger.debug(f"[InformationBottleneck] Forward - Input shape: {x.shape}")

        # Main path: squeeze -> LeakyReLU -> expand -> LeakyReLU
        squeezed = self.squeeze(x)
        logger.debug(f"[InformationBottleneck] After squeeze: {squeezed.shape}")

        activated_squeezed = self.leaky_relu(squeezed)
        expanded = self.expand(activated_squeezed)
        logger.debug(f"[InformationBottleneck] After expand: {expanded.shape}")

        main_output = self.leaky_relu(expanded)

        # Residual connection with alpha_bn weight
        residual = self.alpha_bn * x
        output = main_output + residual

        logger.debug(f"[InformationBottleneck] Output shape: {output.shape}, alpha_bn={self.alpha_bn}")
        return output

    def set_alpha_bn(self, alpha_bn: float):
        old_alpha = self.alpha_bn
        self.alpha_bn = alpha_bn
        logger.info(f"[InformationBottleneck] alpha_bn changed: {old_alpha:.3f} -> {alpha_bn:.3f}")