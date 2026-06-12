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

        # 2. Normalize to unit length for cosine similarity
        z_anchor = F.normalize(z_anchor, p=2, dim=1)
        z_positive = F.normalize(z_positive, p=2, dim=1)

        # 3. Compute similarities
        # Positive similarity: [B]
        sim_positive = torch.sum(z_anchor * z_positive, dim=1)

        # Negative similarities: [B, B]
        # Each row i: similarities between anchor_i and all other samples
        sim_matrix = torch.matmul(z_anchor, z_anchor.t())  # [B, B]

        # 4. Create mask to exclude self-similarity
        # We don't want anchor_i to be compared with itself as a negative
        mask = torch.eye(batch_size, device=z_anchor.device).bool()
        # Use -1e4 instead of -1e4 for float16/AMP compatibility (float16 max is ~65504)
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)

        # 5. Identify hard negatives and compute weights
        neg_weights = self.compute_hard_negative_weights(sim_matrix)  # [B, B]

        # 6. Compute InfoNCE loss
        # Numerator: exp(sim_positive / �)
        numerator = torch.exp(sim_positive / self.temperature)  # [B]

        # Denominator: exp(sim_positive / �) + � w_neg�exp(sim_negative / �)
        # First, compute weighted exp of negative similarities
        weighted_neg_exp = neg_weights * torch.exp(sim_matrix / self.temperature)  # [B, B]
        sum_neg_exp = torch.sum(weighted_neg_exp, dim=1)  # [B]

        denominator = numerator + sum_neg_exp  # [B]

        # Loss: -log(numerator / denominator)
        loss = -torch.log(numerator / (denominator + 1e-8))

        # Average over batch
        loss = torch.mean(loss)

        logger.debug(f"[ContrastiveLoss] Loss: {loss.item():.4f}")
        return loss

    def temporal_pool(self, z_c: torch.Tensor) -> torch.Tensor:
        return torch.mean(z_c, dim=2)

    def compute_hard_negative_weights(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        # Initialize all weights to 1.0
        weights = torch.ones_like(sim_matrix)

        # Find hard negatives: similarity > threshold
        hard_negatives = sim_matrix > self.hard_neg_threshold

        # Set hard negative weights to higher value
        weights[hard_negatives] = self.hard_neg_weight

        return weights


class ContrastivePairGenerator:


    def __init__(self, dataset):
        self.dataset = dataset

    def generate_pairs(
        self,
        audio_paths: list,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # This would be implemented in the DataLoader
        # by loading each audio twice with different shifts
        # For now, this is a placeholder structure
        raise NotImplementedError("Implement in DataLoader with double loading")


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

        # Temporal pooling
        z_anchor = torch.mean(z_c_anchor, dim=2)  # [B, 512]
        z_positive = torch.mean(z_c_positive, dim=2)  # [B, 512]

        # Normalize
        z_anchor = F.normalize(z_anchor, p=2, dim=1)
        z_positive = F.normalize(z_positive, p=2, dim=1)

        # Concatenate anchor and positive: [2B, 512]
        z_all = torch.cat([z_anchor, z_positive], dim=0)

        # Compute similarity matrix: [2B, 2B]
        sim_matrix = torch.matmul(z_all, z_all.t()) / self.temperature

        # Create labels: for each i, its positive is at position i + batch_size (or i - batch_size)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),  # Positives for first half
            torch.arange(0, batch_size)  # Positives for second half
        ], dim=0).to(z_all.device)

        # Mask out self-similarity (use -1e4 for float16/AMP compatibility)
        mask = torch.eye(2 * batch_size, device=z_all.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)

        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(
        self,
        z_c_anchor: torch.Tensor,
        z_c_positive: torch.Tensor
    ) -> torch.Tensor:
        # Pool and normalize
        z_anchor = F.normalize(torch.mean(z_c_anchor, dim=2), dim=1)
        z_positive = F.normalize(torch.mean(z_c_positive, dim=2), dim=1)

        batch_size = z_anchor.shape[0]

        # Concatenate
        representations = torch.cat([z_anchor, z_positive], dim=0)  # [2B, 512]

        # Similarity matrix
        similarity_matrix = torch.matmul(representations, representations.t())  # [2B, 2B]

        # Create positive pair mask
        # For i in [0, B): positive is at i + B
        # For i in [B, 2B): positive is at i - B
        mask = torch.zeros(2 * batch_size, 2 * batch_size, device=z_anchor.device)
        for i in range(batch_size):
            mask[i, i + batch_size] = 1
            mask[i + batch_size, i] = 1

        # Remove self-similarity
        diag_mask = torch.eye(2 * batch_size, device=z_anchor.device)
        similarity_matrix = similarity_matrix * (1 - diag_mask)

        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Compute loss
        # For each sample, the positive should have highest similarity
        exp_sim = torch.exp(similarity_matrix)
        sum_exp = torch.sum(exp_sim, dim=1, keepdim=True)

        log_prob = similarity_matrix - torch.log(sum_exp)
        loss = -torch.sum(log_prob * mask) / (2 * batch_size)

        return loss


# -----------------------------------------------------------------------------
# Consistency regularization
# -----------------------------------------------------------------------------


class ConsistencyLoss(nn.Module):
    """Consistency regularization loss.

    The training stack expects a callable named `ConsistencyLoss` (see
    `training/losses.py`) with the signature:

        loss = consistency(z_c, mel_spec, encoder)

    This implementation is intentionally defensive: it attempts to obtain a
    second content representation by re-encoding `mel_spec` with `encoder` and
    then penalizes the discrepancy between the two representations.

    If `encoder` cannot be called in the expected way (or returns an
    unexpected structure), the loss safely falls back to 0.0 rather than
    crashing training.
    """

    def __init__(self, loss_type: str = "mse", detach_teacher: bool = True):
        super().__init__()
        self.loss_type = loss_type
        self.detach_teacher = detach_teacher

    @staticmethod
    def _extract_z(output):
        """Extract a tensor representation from common encoder outputs."""
        if torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            # Prefer common keys
            for k in ("z_c", "content", "repr", "representation", "features"):
                if k in output and torch.is_tensor(output[k]):
                    return output[k]
            # Otherwise: first tensor value
            for v in output.values():
                if torch.is_tensor(v):
                    return v
        if isinstance(output, (list, tuple)) and len(output) > 0:
            for item in output:
                if torch.is_tensor(item):
                    return item
        raise TypeError("Could not extract tensor representation from encoder output")

    @staticmethod
    def _temporal_pool(x: torch.Tensor) -> torch.Tensor:
        # Supports [B, C, T] or [B, D]
        return x.mean(dim=2) if x.dim() == 3 else x

    def forward(self, z_c: torch.Tensor, mel_spec: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        try:
            # Student representation
            z_student = z_c

            # Teacher / consistency target: re-encode mel
            with torch.no_grad() if self.detach_teacher else torch.enable_grad():
                enc_out = encoder(mel_spec)
            z_teacher = self._extract_z(enc_out)
            if self.detach_teacher:
                z_teacher = z_teacher.detach()

            # If shapes differ, pool to [B, D] and compare
            z_s = self._temporal_pool(z_student)
            z_t = self._temporal_pool(z_teacher)

            # Align feature dim if needed (best-effort)
            if z_s.shape != z_t.shape:
                # Try to align last dim via truncation (safe and deterministic)
                d = min(z_s.shape[-1], z_t.shape[-1])
                z_s = z_s[..., :d]
                z_t = z_t[..., :d]

            if self.loss_type.lower() == "cosine":
                z_s = F.normalize(z_s, p=2, dim=-1)
                z_t = F.normalize(z_t, p=2, dim=-1)
                return (1.0 - (z_s * z_t).sum(dim=-1)).mean()

            # Default: MSE
            return F.mse_loss(z_s, z_t)

        except Exception as e:
            logger.warning(f"[ConsistencyLoss] Falling back to zero loss due to: {e}")
            return torch.zeros((), device=z_c.device)