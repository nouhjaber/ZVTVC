"""
Loss Functions for Prosody Encoder
Includes reconstruction, consistency, and smoothness losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for explicit features
    L_recon = w_f0 * L_f0 + w_energy * L_energy + w_voicing * L_voicing + w_rhythm * L_rhythm
    """

    def __init__(
        self,
        f0_weight: float = 1.0,
        energy_weight: float = 1.0,
        voicing_weight: float = 1.0,
        rhythm_weight: float = 1.0,
        voicing_loss_type: str = "bce",  # 'bce' or 'mse'
    ):
        """
        Initialize reconstruction loss
        """
        super().__init__()

        self.f0_weight = f0_weight
        self.energy_weight = energy_weight
        self.voicing_weight = voicing_weight
        self.rhythm_weight = rhythm_weight
        self.voicing_loss_type = voicing_loss_type

    def forward(
        self,
        reconstructions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Compute reconstruction loss
        # Split targets
        target_f0 = targets[:, 0:1, :]  # [B, 1, T]
        target_energy = targets[:, 1:2, :]  # [B, 1, T]
        target_voicing = targets[:, 2:3, :]  # [B, 1, T]
        target_rhythm = targets[:, 3:4, :]  # [B, 1, T]

        # F0 loss (MSE)
        loss_f0 = F.mse_loss(reconstructions["f0"], target_f0)

        # Energy loss (MSE)
        loss_energy = F.mse_loss(reconstructions["energy"], target_energy)

        # Voicing loss (BCE or MSE)
        if self.voicing_loss_type == "bce":
            # Note: reconstructions["voicing"] should already be sigmoid output
            loss_voicing = F.binary_cross_entropy(
                reconstructions["voicing"],
                target_voicing,
            )
        else:
            loss_voicing = F.mse_loss(reconstructions["voicing"], target_voicing)

        # Rhythm loss (MSE)
        loss_rhythm = F.mse_loss(reconstructions["rhythm"], target_rhythm)

        # Weighted sum
        total_loss = (
            self.f0_weight * loss_f0 +
            self.energy_weight * loss_energy +
            self.voicing_weight * loss_voicing +
            self.rhythm_weight * loss_rhythm
        )

        # Return detailed losses
        losses = {
            "reconstruction": total_loss,
            "f0": loss_f0,
            "energy": loss_energy,
            "voicing": loss_voicing,
            "rhythm": loss_rhythm,
        }

        return losses


class ConsistencyLoss(nn.Module):
    """
    Consistency loss for augmentation robustness
    L_consist = MSE(refined_clean, refined_augmented)
    """

    def __init__(self, distance_metric: str = "mse"):
        """
        Initialize consistency loss

        Args:
            distance_metric: Distance metric ('mse', 'l1', or 'cosine')
        """
        super().__init__()
        self.distance_metric = distance_metric

    def forward(
        self,
        refined_clean: torch.Tensor,
        refined_augmented: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute consistency loss
        """
        if self.distance_metric == "mse":
            loss = F.mse_loss(refined_clean, refined_augmented)
        elif self.distance_metric == "l1":
            loss = F.l1_loss(refined_clean, refined_augmented)
        elif self.distance_metric == "cosine":
            # Cosine similarity loss
            cos_sim = F.cosine_similarity(
                refined_clean.flatten(1),
                refined_augmented.flatten(1),
                dim=1
            ).mean()
            loss = 1 - cos_sim  # Convert to loss
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return loss


class SmoothnessLoss(nn.Module):
    """
    Smoothness loss for temporal continuity
    L_smooth = mean(|refined[:, :, t+1] - refined[:, :, t]|)
    """

    def __init__(self, order: int = 1):
        """
        Initialize smoothness loss

        Args:
            order: Order of derivative (1 or 2)
        """
        super().__init__()
        self.order = order

    def forward(self, refined: torch.Tensor) -> torch.Tensor:
        if self.order == 1:
            # First-order derivative (frame-to-frame difference)
            diff = refined[:, :, 1:] - refined[:, :, :-1]
            loss = torch.mean(torch.abs(diff))
        elif self.order == 2:
            # Second-order derivative (acceleration)
            diff1 = refined[:, :, 1:] - refined[:, :, :-1]
            diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]
            loss = torch.mean(torch.abs(diff2))
        else:
            raise ValueError(f"Unsupported order: {self.order}")

        return loss


class ProsodyLoss(nn.Module):
    """
    Combined loss for prosody encoder training
    L_total = L_recon + λ_consist × L_consist + λ_smooth × L_smooth
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        consistency_weight: float = 0.5,
        smoothness_weight: float = 0.1,
        f0_weight: float = 1.0,
        energy_weight: float = 1.0,
        voicing_weight: float = 1.0,
        rhythm_weight: float = 1.0,
        voicing_loss_type: str = "bce",
        smoothness_order: int = 1,
    ):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight

        # Component losses
        self.reconstruction_loss = ReconstructionLoss(
            f0_weight=f0_weight,
            energy_weight=energy_weight,
            voicing_weight=voicing_weight,
            rhythm_weight=rhythm_weight,
            voicing_loss_type=voicing_loss_type,
        )

        self.consistency_loss = ConsistencyLoss(distance_metric="mse")
        self.smoothness_loss = SmoothnessLoss(order=smoothness_order)

    def forward(
        self,
        refined: torch.Tensor,
        reconstructions: Optional[Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        refined_augmented: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        total_loss = torch.zeros(1, device=refined.device, dtype=refined.dtype).squeeze()
        losses = {}

        # Reconstruction loss
        if reconstructions is not None and self.reconstruction_weight > 0:
            recon_losses = self.reconstruction_loss(reconstructions, targets)
            losses.update(recon_losses)
            total_loss += self.reconstruction_weight * recon_losses["reconstruction"]

        # Consistency loss
        if refined_augmented is not None and self.consistency_weight > 0:
            consist_loss = self.consistency_loss(refined, refined_augmented)
            losses["consistency"] = consist_loss
            total_loss += self.consistency_weight * consist_loss

        # Smoothness loss
        if self.smoothness_weight > 0:
            smooth_loss = self.smoothness_loss(refined)
            losses["smoothness"] = smooth_loss
            total_loss += self.smoothness_weight * smooth_loss

        losses["total"] = total_loss

        return losses


def test_losses():
    """Test loss functions"""
    print("Testing Loss Functions...")

    batch_size = 4
    seq_length = 100

    # Create dummy data
    targets = torch.randn(batch_size, 4, seq_length)
    reconstructions = {
        "f0": torch.randn(batch_size, 1, seq_length),
        "energy": torch.randn(batch_size, 1, seq_length),
        "voicing": torch.sigmoid(torch.randn(batch_size, 1, seq_length)),
        "rhythm": torch.randn(batch_size, 1, seq_length),
    }
    refined = torch.randn(batch_size, 32, seq_length)
    refined_augmented = refined + torch.randn_like(refined) * 0.1

    # Test reconstruction loss
    print("\nTesting Reconstruction Loss...")
    recon_loss = ReconstructionLoss()
    losses = recon_loss(reconstructions, targets)
    print(f"Reconstruction losses: {[(k, v.item()) for k, v in losses.items()]}")

    # Test consistency loss
    print("\nTesting Consistency Loss...")
    consist_loss = ConsistencyLoss()
    loss = consist_loss(refined, refined_augmented)
    print(f"Consistency loss: {loss.item():.4f}")

    # Test smoothness loss
    print("\nTesting Smoothness Loss...")
    smooth_loss = SmoothnessLoss(order=1)
    loss = smooth_loss(refined)
    print(f"Smoothness loss: {loss.item():.4f}")

    # Test combined loss
    print("\nTesting Combined Prosody Loss...")
    prosody_loss = ProsodyLoss()
    losses = prosody_loss(refined, reconstructions, targets, refined_augmented)
    print(f"Total loss: {losses['total'].item():.4f}")
    print(f"All losses: {[(k, v.item()) for k, v in losses.items()]}")

    print("\nTest passed!")


if __name__ == "__main__":
    test_losses()