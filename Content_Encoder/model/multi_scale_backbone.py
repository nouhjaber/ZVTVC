import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

from model.encoder import EncoderBlock

logger = logging.getLogger(__name__)

class MultiScaleEncoder(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.channels = channels
        
        # Fine Path: 4 blocks with dilation=1
        self.fine_path = nn.Sequential(
            EncoderBlock(channels, dilation=1),
            EncoderBlock(channels, dilation=1),
            EncoderBlock(channels, dilation=1),
            EncoderBlock(channels, dilation=1)
        )
        
        # Medium Path: 4 blocks with pattern [1, 2, 4, 1]
        self.medium_path = nn.Sequential(
            EncoderBlock(channels, dilation=1),
            EncoderBlock(channels, dilation=2),
            EncoderBlock(channels, dilation=4),
            EncoderBlock(channels, dilation=1)
        )
        
        # Coarse Path: 5 blocks with pyramid pattern [1, 4, 16, 4, 1]
        self.coarse_path = nn.Sequential(
            EncoderBlock(channels, dilation=1),
            EncoderBlock(channels, dilation=4),
            EncoderBlock(channels, dilation=16),
            EncoderBlock(channels, dilation=4),
            EncoderBlock(channels, dilation=1)
        )

        logger.info(f"[MultiScaleEncoder] Initialized with channels={channels}")
        logger.info(f"[MultiScaleEncoder] Fine path: 4 blocks [dilation=1]")
        logger.info(f"[MultiScaleEncoder] Medium path: 4 blocks [dilation=1,2,4,1]")
        logger.info(f"[MultiScaleEncoder] Coarse path: 5 blocks [dilation=1,4,16,4,1]")
    
    def forward(self, x):
        logger.debug(f"[MultiScaleEncoder] Forward - Input: {x.shape}")

        # Process input through all three paths in parallel
        fine_out = self.fine_path(x)
        logger.debug(f"[MultiScaleEncoder] Fine path output: {fine_out.shape}")

        medium_out = self.medium_path(x)
        logger.debug(f"[MultiScaleEncoder] Medium path output: {medium_out.shape}")

        coarse_out = self.coarse_path(x)
        logger.debug(f"[MultiScaleEncoder] Coarse path output: {coarse_out.shape}")

        return fine_out, medium_out, coarse_out