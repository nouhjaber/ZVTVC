"""
Multi-Task Loss for Content Encoder
4-STAGE TRAINING: Bootstrap → Self-Shifting → Bottleneck Adaptation → Full Adversarial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging
from training.distillation import DistillationLoss
from training.contrastive import ContrastiveLoss
from training.consistency import ConsistencyLoss

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Combines 5 losses with FIXED weights per stage.
    
    4-STAGE DESIGN:
      Stage 0 (Bootstrap): Learn phonetic content from teachers. No adversarial.
      Stage 1 (Self-Shifting): Introduce adversarial with warmup at α=0.3 bottleneck.
      Stage 2 (Bottleneck Adapt): Tighten bottleneck to α=0.2, keep adversarial clamped
               so encoder stabilizes in new representation space. Short stage (~30k iters).
      Stage 3 (Full Adversarial): Fresh adversarial warmup from zero in the stable α=0.2
               space. Speaker classifier relearns from scratch — no explosion.
    """

    def __init__(
        self,
        distillation_loss: DistillationLoss,
        contrastive_loss: ContrastiveLoss,
        consistency_loss: ConsistencyLoss,
        num_phoneme_classes: int = 78,
        num_speakers: int = 100
    ):
        super().__init__()

        # Loss modules
        self.distillation = distillation_loss
        self.contrastive = contrastive_loss
        self.consistency = consistency_loss

        self.num_phoneme_classes = num_phoneme_classes
        self.num_speakers = num_speakers

        # Dummy parameter so state_dict loading doesn't break
        # (old checkpoints have log_sigma in state_dict)
        self.log_sigma = nn.Parameter(torch.zeros(5), requires_grad=False)

        # Fixed loss weights per stage
        self.stage_weights = {
            0: {'distill': 1.0, 'phoneme': 0.5, 'speaker': 0.0, 'contrast': 0.1, 'consist': 0.01},
            1: {'distill': 1.0, 'phoneme': 0.3, 'speaker': 0.3, 'contrast': 0.2, 'consist': 0.05},
            2: {'distill': 1.0, 'phoneme': 0.25, 'speaker': 0.3, 'contrast': 0.2, 'consist': 0.05},
            3: {'distill': 1.0, 'phoneme': 0.25, 'speaker': 0.3, 'contrast': 0.2, 'consist': 0.05},
        }

        # Current adversarial parameters
        self.lambda_adv = 0.0
        self.lambda_grl = 0.0

        logger.info(f"[MultiTaskLoss] Initialized with FIXED weights (4-stage)")
        logger.info(f"[MultiTaskLoss] {num_phoneme_classes} phonemes, {num_speakers} speakers")

    def forward(
        self,
        z_c: torch.Tensor,
        phoneme_logits: torch.Tensor,
        speaker_logits: torch.Tensor,
        phoneme_labels: torch.Tensor,
        phoneme_confidence: torch.Tensor,
        speaker_labels: torch.Tensor,
        mel_spec: torch.Tensor,
        z_c_positive: torch.Tensor,
        audio_or_mel: torch.Tensor,
        encoder: nn.Module,
        iteration: int,
        stage: int,
        waveform: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:

        losses = {}
        weights = self.stage_weights.get(stage, self.stage_weights[0])

        # 1. DISTILLATION LOSS - pass waveform for mHuBERT
        distill_dict = self.distillation(z_c, audio_or_mel, iteration, waveform=waveform)
        loss_distill = distill_dict['total']
        losses['distill'] = loss_distill
        losses['distill_whisper'] = distill_dict['whisper']
        losses['distill_mhubert'] = distill_dict['mhubert']
        losses['distill_ema'] = distill_dict['ema']

        # 2. PHONEME LOSS
        loss_phoneme = self.compute_phoneme_loss(
            phoneme_logits, phoneme_labels, phoneme_confidence
        )
        losses['phoneme'] = loss_phoneme

        # 3. SPEAKER ADVERSARIAL LOSS
        if stage > 0 and self.lambda_adv > 0:
            loss_speaker = self.compute_speaker_loss(speaker_logits, speaker_labels, stage)
        else:
            loss_speaker = torch.tensor(0.0, device=z_c.device)
        losses['speaker'] = loss_speaker

        # 4. CONTRASTIVE LOSS
        loss_contrast = self.contrastive(z_c, z_c_positive)
        losses['contrast'] = loss_contrast

        # 5. CONSISTENCY LOSS
        if self.training:
            z_c_aug = encoder(mel_spec)  # Different due to dropout
            loss_consist = F.mse_loss(z_c, z_c_aug.detach())
        else:
            loss_consist = torch.tensor(0.0, device=z_c.device)
        losses['consist'] = loss_consist

        # Weighted sum
        total_loss = (
            weights['distill'] * loss_distill +
            weights['phoneme'] * loss_phoneme +
            weights['speaker'] * (self.lambda_adv * loss_speaker) +
            weights['contrast'] * loss_contrast +
            weights['consist'] * loss_consist
        )

        losses['total'] = total_loss

        # Keep sigma keys for compatibility
        losses['sigma_distill'] = torch.tensor(1.0)
        losses['sigma_phoneme'] = torch.tensor(1.0)
        losses['sigma_speaker'] = torch.tensor(1.0)
        losses['sigma_contrast'] = torch.tensor(1.0)
        losses['sigma_consist'] = torch.tensor(1.0)

        return losses

    def compute_phoneme_loss(
        self,
        phoneme_logits: torch.Tensor,
        phoneme_labels: torch.Tensor,
        phoneme_confidence: torch.Tensor
    ) -> torch.Tensor:
        B, C, T = phoneme_logits.shape
        logits_flat = phoneme_logits.transpose(1, 2).reshape(-1, C)
        labels_flat = phoneme_labels.reshape(-1)
        conf_flat = phoneme_confidence.reshape(-1)
        
        labels_flat = labels_flat.clamp(0, C - 1)

        weights = ((conf_flat - 0.3) / 0.4).clamp(0.0, 1.0)

        ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction='none',
                                  label_smoothing=0.1)
        
        ce_loss = torch.where(torch.isfinite(ce_loss), ce_loss, torch.zeros_like(ce_loss))

        weighted_loss = ce_loss * weights
        
        denom = weights.sum().clamp(min=1.0)
        return weighted_loss.sum() / denom

    def compute_speaker_loss(
        self,
        speaker_logits: torch.Tensor,
        speaker_labels: torch.Tensor,
        stage: int = 2
    ) -> torch.Tensor:
        loss = F.cross_entropy(speaker_logits, speaker_labels)

        # Stage 2: Hard clamp at 10.0. The speaker classifier can't work through
        # the newly-tightened bottleneck (α=0.2) and produces CE loss 50-100+.
        # We clamp to prevent destabilization while the encoder adapts.
        #
        # Stage 3: Raised clamp at 20.0. The classifier is re-learning from scratch
        # via fresh warmup in the now-stable α=0.2 space. We allow higher values
        # so it can provide real gradient signal, but still cap extreme outliers.
        if stage <= 2:
            return torch.clamp(loss, max=10.0)
        else:
            return torch.clamp(loss, max=20.0)

    def update_adversarial_params(self, lambda_adv: float, lambda_grl: float):
        self.lambda_adv = lambda_adv
        self.lambda_grl = lambda_grl


class LossScheduler:
    """Schedule adversarial loss parameters per stage."""

    def __init__(self, stage: int = 0):
        self.stage = stage
        # Stage 0: No adversarial
        # Stage 1: Warmup adversarial 0→0.02 / 0→0.50
        # Stage 2: Keep Stage 1 endpoint, clamped (bottleneck adaptation)
        # this stage faild 
        # Stage 3: FRESH warmup to STRONGER values than Stage 1.
        #          The encoder is now stable at α=0.2, and the speaker classifier
        #          rebuilds from scratch — so it can handle stronger pressure
        #          without the explosion we saw when jumping straight from Stage 1.
        #          λ_adv: 2× Stage 1, λ_grl: 1.6× Stage 1, longer warmup (8k iters) 
        self.stage_configs = {
            0: {'lambda_adv_start': 0.0, 'lambda_adv_end': 0.0, 'lambda_grl_start': 0.0, 'lambda_grl_end': 0.0, 'warmup': 0},
            1: {'lambda_adv_start': 0.0, 'lambda_adv_end': 0.02, 'lambda_grl_start': 0.0, 'lambda_grl_end': 0.50, 'warmup': 5000},
            2: {'lambda_adv_start': 0.02, 'lambda_adv_end': 0.02, 'lambda_grl_start': 0.50, 'lambda_grl_end': 0.50, 'warmup': 0},
            3: {'lambda_adv_start': 0.0, 'lambda_adv_end': 0.04, 'lambda_grl_start': 0.0, 'lambda_grl_end': 0.80, 'warmup': 8000}, 
        }

    def get_params(self, iteration: int):
        """Get adversarial params. iteration should be STAGE-RELATIVE."""
        config = self.stage_configs.get(self.stage, self.stage_configs[0])
        warmup = config['warmup']

        if warmup > 0 and iteration < warmup:
            progress = iteration / warmup
            lambda_adv = config['lambda_adv_start'] + progress * (config['lambda_adv_end'] - config['lambda_adv_start'])
            lambda_grl = config['lambda_grl_start'] + progress * (config['lambda_grl_end'] - config['lambda_grl_start'])
        else:
            lambda_adv = config['lambda_adv_end']
            lambda_grl = config['lambda_grl_end']

        return lambda_adv, lambda_grl


class MetricsTracker:
    """Helper class to track and log training metrics."""

    def __init__(self):
        self.history = {
            'total_loss': [],
            'distill': [],
            'phoneme': [],
            'speaker': [],
            'contrast': [],
            'consist': []
        }

    def update(self, losses: Dict[str, torch.Tensor]):
        for key in self.history.keys():
            if key in losses:
                value = losses[key].item() if torch.is_tensor(losses[key]) else losses[key]
                self.history[key].append(value)

    def get_recent_average(self, key: str, window: int = 100) -> float:
        if key not in self.history or len(self.history[key]) == 0:
            return 0.0
        recent = self.history[key][-window:]
        return sum(recent) / len(recent)