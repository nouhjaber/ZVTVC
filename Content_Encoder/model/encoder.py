import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1,
                 dilation: int = 1,
                 bias: bool = True,
                 groups: int = 1,
                 padding_mode: str = "zeros"):
        
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # This ensures output at time t only depends on t, t-1, t-2, ..., t-(kernel_size-1)
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            groups=groups,
            padding=0,
            padding_mode=padding_mode
        )

        logger.debug(f"[CausalConv1d] Initialized: in={in_channels}, out={out_channels}, "
                    f"kernel={kernel_size}, stride={stride}, dilation={dilation}, padding={self.padding}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"[CausalConv1d] Forward - Input: {x.shape}")
        x_padded = F.pad(x, (self.padding, 0), mode='constant', value=0)
        logger.debug(f"[CausalConv1d] After padding: {x_padded.shape}")
        output = self.conv(x_padded)
        logger.debug(f"[CausalConv1d] Output: {output.shape}")
        return output


class EncoderBlock(nn.Module):
    def __init__(self, channels: int = 256, kernel_size: int = 3, 
                 dilation: int = 1, dropout_rate: float = 0.1, 
                 leaky_relu_slope: float = 0.2):
        
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.leaky_relu_slope = leaky_relu_slope
        
        # First causal convolution layer
        self.conv1 = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding_mode='zeros'
        )
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.layer_norm1 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Second causal convolution layer
        self.conv2 = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding_mode='zeros'
        )
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.layer_norm2 = nn.LayerNorm(channels)

        logger.info(f"[EncoderBlock] Initialized: channels={channels}, kernel_size={kernel_size}, "
                   f"dilation={dilation}, dropout={dropout_rate}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"[EncoderBlock] Forward - Input: {x.shape}, dilation={self.dilation}")
        residual = x

        # Input shape: [Batch, 256, Time]
        out = self.conv1(x)
        out = self.leaky_relu1(out)

        out = out.transpose(1, 2)  # [Batch, Time, 256]
        out = self.layer_norm1(out)
        out = out.transpose(1, 2)  # [Batch, 256, Time]
        out = self.dropout(out)
        logger.debug(f"[EncoderBlock] After first conv+norm: {out.shape}")

        # Second causal convolution layer
        out = self.conv2(out)
        out = self.leaky_relu2(out)

        # Apply LayerNorm with transpose
        out = out.transpose(1, 2)  # [Batch, Time, 256]
        out = self.layer_norm2(out)
        out = out.transpose(1, 2)  # [Batch, 256, Time]

        # Add residual connection
        out = out + residual
        logger.debug(f"[EncoderBlock] Output (after residual): {out.shape}")

        return out

