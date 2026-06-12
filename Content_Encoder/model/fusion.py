import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from model.multi_scale_backbone import MultiScaleEncoder

logger = logging.getLogger(__name__)

class HierarchicalFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.w_alpha = nn.Parameter(torch.tensor(0.0))
        self.w_beta = nn.Parameter(torch.tensor(0.0))

        logger.info(f"[HierarchicalFusion] Initialized with learnable weights w_alpha={self.w_alpha.item():.3f}, w_beta={self.w_beta.item():.3f}")

    def forward(self, fine, medium, coarse):
        logger.debug(f"[HierarchicalFusion] Forward - Fine: {fine.shape}, Medium: {medium.shape}, Coarse: {coarse.shape}")

        # Fuse Fine + Medium:
        alpha = torch.sigmoid(self.w_alpha)
        fuse_1 = alpha * fine + (1 - alpha) * medium
        logger.debug(f"[HierarchicalFusion] alpha={alpha.item():.3f}, fuse_1: {fuse_1.shape}")

        # Fuse result + Coarse:
        beta = torch.sigmoid(self.w_beta)
        fuse_2 = beta * fuse_1 + (1 - beta) * coarse
        logger.debug(f"[HierarchicalFusion] beta={beta.item():.3f}, fuse_2: {fuse_2.shape}")

        return fuse_2
    
    def get_fusion_weights(self):
        with torch.no_grad():
            alpha = torch.sigmoid(self.w_alpha).item()
            beta = torch.sigmoid(self.w_beta).item()
        return alpha, beta

        


