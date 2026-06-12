"""
Conditional Flow Matching (CFM) Training Logic
Implements the flow matching algorithm for mel-spectrogram generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class ConditionalFlowMatching:
    """
    Conditional Flow Matching for training.

    Implements linear interpolation between noise and data:
        x_t = (1 - t) * x0 + t * x1

    Where:
        - x0 ~ N(0, I) is Gaussian noise
        - x1 is the target mel-spectrogram
        - t ~ Uniform(0, 1) is the time
    """

    def __init__(
        self,
        time_sampling: str = "uniform",
        eps: float = 1e-5
    ):
        self.time_sampling = time_sampling
        self.eps = eps

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.time_sampling == "uniform":
            # Uniform sampling: t ~ Uniform(0, 1)
            t = torch.rand(batch_size, device=device)

        elif self.time_sampling == "logit_normal":
            # Logit-normal sampling (concentrates around t=0.5)
            # More stable for some cases
            t = torch.randn(batch_size, device=device)
            t = torch.sigmoid(t)

        else:
            raise ValueError(f"Unknown time sampling: {self.time_sampling}")

        # Ensure t is in (eps, 1-eps) for numerical stability
        t = torch.clamp(t, self.eps, 1.0 - self.eps)

        return t

    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        # Expand t to match dimensions: [B] -> [B, 1, 1]
        t = t.view(-1, 1, 1)

        # Linear interpolation
        x_t = (1 - t) * x0 + t * x1

        return x_t

    def compute_velocity_target(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the target velocity.

        For linear interpolation: v = dx_t/dt = x1 - x0 (constant!)
        """
        return x1 - x0


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching Loss: MSE between predicted and target velocity.

    Loss = E[||v_pred - v_target||^2]

    Where:
        - v_pred = model(x_t, t, conditions)
        - v_target = x1 - x0
    """

    def __init__(
        self,
        cfm: ConditionalFlowMatching,
        time_weighting: bool = False
    ):
        super().__init__()
        self.cfm = cfm
        self.time_weighting = time_weighting

    def forward(
        self,
        model: nn.Module,
        mel_target: torch.Tensor,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        batch_size = mel_target.shape[0]
        device = mel_target.device

        # 1. Sample time steps
        t = self.cfm.sample_time(batch_size, device)

        # 2. Sample noise (starting point)
        x0 = torch.randn_like(mel_target)

        # 3. Target mel (ending point)
        x1 = mel_target

        # 4. Interpolate to get x_t
        x_t = self.cfm.interpolate(x0, x1, t)

        # 5. Compute target velocity
        v_target = self.cfm.compute_velocity_target(x0, x1)

        # 6. Predict velocity with model
        v_pred = model(x_t, t, content, prosody, timbre)

        # 7. Compute MSE loss
        loss = F.mse_loss(v_pred, v_target, reduction='none')

        # Optional: Time-dependent weighting
        if self.time_weighting:
            # Higher weight near t=1 (close to data)
            weight = 1.0 / (1.0 - t.view(-1, 1, 1) + 1e-5)
            loss = loss * weight

        # Average over all dimensions
        loss = loss.mean()

        if return_components:
            return {
                'loss': loss,
                'v_pred': v_pred,
                'v_target': v_target,
                'x_t': x_t,
                't': t,
                'x0': x0,
                'x1': x1
            }
        else:
            return {'loss': loss}


class ClassifierFreeGuidance:
    """
    Classifier-Free Guidance for conditional generation.

    During training: Randomly drop conditions with probability p_uncond
    During inference: v_guided = v_uncond + s * (v_cond - v_uncond)
    """

    def __init__(
        self,
        p_uncond: float = 0.1,
        guidance_scale: float = 1.5
    ):
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale

    def drop_conditions(
        self,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not training or self.p_uncond == 0.0:
            return content, prosody, timbre

        batch_size = content.shape[0]
        device = content.device

        # Sample which samples to drop
        mask = torch.rand(batch_size, device=device) < self.p_uncond
        mask = mask.view(-1, 1, 1)  # [B, 1, 1]

        # Zero out conditions
        content_dropped = torch.where(mask, torch.zeros_like(content), content)
        prosody_dropped = torch.where(mask, torch.zeros_like(prosody), prosody)

        # For timbre, use learned null embedding (or zeros)
        mask_timbre = mask.squeeze(-1)  # [B, 1]
        timbre_dropped = torch.where(mask_timbre, torch.zeros_like(timbre), timbre)

        return content_dropped, prosody_dropped, timbre_dropped

    def apply_guidance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor
    ) -> torch.Tensor:
        # Apply classifier-free guidance during inference.
        if self.guidance_scale == 1.0:
            # No guidance, just return conditional prediction
            return model(x_t, t, content, prosody, timbre)

        # Conditional prediction
        v_cond = model(x_t, t, content, prosody, timbre)

        # Unconditional prediction (with zeroed conditions)
        content_null = torch.zeros_like(content)
        prosody_null = torch.zeros_like(prosody)
        timbre_null = torch.zeros_like(timbre)

        v_uncond = model(x_t, t, content_null, prosody_null, timbre_null)

        # Apply guidance
        v_guided = v_uncond + self.guidance_scale * (v_cond - v_uncond)

        return v_guided


# ==============================================================================
# MULTI-RESOLUTION STFT LOSS (Auxiliary Loss for Mel Quality)
# ==============================================================================

class STFTLoss(nn.Module):
    """
    Single-resolution STFT loss.
    
    Computes:
    - Spectral convergence loss (relative difference)
    - Log magnitude loss (absolute difference in log domain)
    """
    
    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_size: int = 1024,
        window: str = "hann"
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        
        # Register window as buffer
        if window == "hann":
            self.register_buffer("window", torch.hann_window(win_size))
        else:
            self.register_buffer("window", torch.ones(win_size))
    
    def stft(self, x: torch.Tensor) -> torch.Tensor:
        # Handle mel input [B, C, T] - treat as pseudo-waveform per channel
        if x.dim() == 3:
            B, C, T = x.shape
            # Flatten channels into batch for STFT
            x = x.reshape(B * C, T)
            
            # Pad to at least fft_size to avoid STFT crash on short sequences
            if x.shape[-1] < self.fft_size:
                pad_amount = self.fft_size - x.shape[-1]
                x = torch.nn.functional.pad(x, (0, pad_amount))
            
            # STFT
            spec = torch.stft(
                x,
                n_fft=self.fft_size,
                hop_length=self.hop_size,
                win_length=self.win_size,
                window=self.window.to(x.device),
                return_complex=True,
                center=True,
                pad_mode='reflect'
            )
            
            # Magnitude
            mag = torch.abs(spec)  # [B*C, F, T']
            
            # Reshape back
            F_bins, T_out = mag.shape[1], mag.shape[2]
            mag = mag.reshape(B, C, F_bins, T_out)
            
            # Average over mel channels for single magnitude
            mag = mag.mean(dim=1)  # [B, F, T']
            
            return mag
        else:
            # Standard waveform input [B, T]
            # Pad to at least fft_size to avoid STFT crash on short sequences
            if x.shape[-1] < self.fft_size:
                pad_amount = self.fft_size - x.shape[-1]
                x = torch.nn.functional.pad(x, (0, pad_amount))
            
            spec = torch.stft(
                x,
                n_fft=self.fft_size,
                hop_length=self.hop_size,
                win_length=self.win_size,
                window=self.window.to(x.device),
                return_complex=True,
                center=True,
                pad_mode='reflect'
            )
            return torch.abs(spec)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute STFT magnitudes
        pred_mag = self.stft(pred)
        target_mag = self.stft(target)
        
        # Spectral convergence loss
        # ||target - pred|| / ||target||
        sc_loss = torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-7)
        
        # Log magnitude loss
        # |log(target) - log(pred)|
        log_pred = torch.log(pred_mag + 1e-7)
        log_target = torch.log(target_mag + 1e-7)
        mag_loss = F.l1_loss(log_pred, log_target)
        
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss.
    
    Computes STFT loss at multiple resolutions to capture both
    fine-grained and coarse spectral details.
    
    Default resolutions: [512, 1024, 2048]
    - 512: Fine temporal resolution, coarse frequency
    - 1024: Balanced
    - 2048: Fine frequency resolution, coarse temporal
    
    Training-only: No impact on inference speed!
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: Optional[List[int]] = None,
        win_sizes: Optional[List[int]] = None,
        window: str = "hann",
        sc_weight: float = 1.0,
        mag_weight: float = 1.0
    ):
        super().__init__()
        
        self.fft_sizes = fft_sizes
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight
        
        # Default hop and win sizes
        if hop_sizes is None:
            hop_sizes = [fft // 4 for fft in fft_sizes]
        if win_sizes is None:
            win_sizes = fft_sizes
        
        # Create STFT loss for each resolution
        self.stft_losses = nn.ModuleList([
            STFTLoss(
                fft_size=fft,
                hop_size=hop,
                win_size=win,
                window=window
            )
            for fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)
        ])
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-resolution STFT loss.
        """
        total_sc_loss = 0.0
        total_mag_loss = 0.0
        
        for stft_loss in self.stft_losses:
            sc_loss, mag_loss = stft_loss(pred, target)
            total_sc_loss += sc_loss
            total_mag_loss += mag_loss
        
        # Average over resolutions
        num_resolutions = len(self.stft_losses)
        total_sc_loss /= num_resolutions
        total_mag_loss /= num_resolutions
        
        # Combined loss
        combined = self.sc_weight * total_sc_loss + self.mag_weight * total_mag_loss
        
        return {
            'stft_loss': combined,
            'sc_loss': total_sc_loss,
            'mag_loss': total_mag_loss
        }