import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_threshold: float = 0.5,
        hard_negative_weight: float = 2.0
    ):
        super().__init__()

        self.temperature = temperature
        self.hard_neg_threshold = hard_negative_threshold
        self.hard_neg_weight = hard_negative_weight

        logger.info(f"[ContrastiveLoss] Initialized - temperature={temperature}, "
                   f"hard_neg_threshold={hard_negative_threshold}, hard_neg_weight={hard_negative_weight}")

    def forward(
        self,
        z_c_anchor: torch.Tensor,
        z_c_positive: torch.Tensor
    ) -> torch.Tensor:
        logger.debug(f"[ContrastiveLoss] Forward - anchor: {z_c_anchor.shape}, positive: {z_c_positive.shape}")
        batch_size = z_c_anchor.shape[0]

        # 1. Temporal pooling - average over time dimension
        z_anchor = self.temporal_pool(z_c_anchor)  # [B, 512]
        z_positive = self.temporal_pool(z_c_positive)  # [B, 512]
        logger.debug(f"[ContrastiveLoss] After pooling - anchor: {z_anchor.shape}, positive: {z_positive.shape}")

        # 2. Normalize to unit length for cosine similarity (with eps for stability)
        z_anchor = F.normalize(z_anchor, p=2, dim=1, eps=1e-8)
        z_positive = F.normalize(z_positive, p=2, dim=1, eps=1e-8)

        # 3. Compute similarities
        sim_positive = torch.sum(z_anchor * z_positive, dim=1)  # [B]
        sim_matrix = torch.matmul(z_anchor, z_anchor.t())  # [B, B]

        # 4. Mask self-similarity
        mask = torch.eye(batch_size, device=z_anchor.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)

        # 5. Hard negative weights
        neg_weights = self.compute_hard_negative_weights(sim_matrix)

        # 6. Compute InfoNCE loss with numerical stability
        # Clamp to prevent exp overflow
        sim_pos_scaled = (sim_positive / self.temperature).clamp(-50, 50)
        sim_mat_scaled = (sim_matrix / self.temperature).clamp(-50, 50)
        
        numerator = torch.exp(sim_pos_scaled)
        weighted_neg_exp = neg_weights * torch.exp(sim_mat_scaled)
        sum_neg_exp = torch.sum(weighted_neg_exp, dim=1)

        denominator = numerator + sum_neg_exp

        # Loss with safety
        ratio = (numerator / (denominator + 1e-8)).clamp(min=1e-8)
        loss = -torch.log(ratio)
        
        # Replace NaN/Inf
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        return torch.mean(loss)

    def temporal_pool(self, z_c: torch.Tensor) -> torch.Tensor:
        return torch.mean(z_c, dim=2)

    def compute_hard_negative_weights(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(sim_matrix)
        hard_negatives = sim_matrix > self.hard_neg_threshold
        weights[hard_negatives] = self.hard_neg_weight
        return weights


class SimCLRLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_c_anchor: torch.Tensor,
        z_c_positive: torch.Tensor
    ) -> torch.Tensor:
        batch_size = z_c_anchor.shape[0]

        z_anchor = torch.mean(z_c_anchor, dim=2)
        z_positive = torch.mean(z_c_positive, dim=2)

        z_anchor = F.normalize(z_anchor, p=2, dim=1, eps=1e-8)
        z_positive = F.normalize(z_positive, p=2, dim=1, eps=1e-8)

        z_all = torch.cat([z_anchor, z_positive], dim=0)
        sim_matrix = torch.matmul(z_all, z_all.t()) / self.temperature

        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ], dim=0).to(z_all.device)

        mask = torch.eye(2 * batch_size, device=z_all.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_c_anchor: torch.Tensor,
        z_c_positive: torch.Tensor
    ) -> torch.Tensor:
        z_anchor = F.normalize(torch.mean(z_c_anchor, dim=2), dim=1, eps=1e-8)
        z_positive = F.normalize(torch.mean(z_c_positive, dim=2), dim=1, eps=1e-8)

        batch_size = z_anchor.shape[0]
        representations = torch.cat([z_anchor, z_positive], dim=0)
        similarity_matrix = torch.matmul(representations, representations.t())

        mask = torch.zeros(2 * batch_size, 2 * batch_size, device=z_anchor.device)
        for i in range(batch_size):
            mask[i, i + batch_size] = 1
            mask[i + batch_size, i] = 1

        diag_mask = torch.eye(2 * batch_size, device=z_anchor.device)
        similarity_matrix = similarity_matrix * (1 - diag_mask)
        similarity_matrix = similarity_matrix / self.temperature

        exp_sim = torch.exp(similarity_matrix.clamp(-50, 50))
        sum_exp = torch.sum(exp_sim, dim=1, keepdim=True)

        log_prob = similarity_matrix - torch.log(sum_exp + 1e-8)
        loss = -torch.sum(log_prob * mask) / (2 * batch_size)

        return loss
