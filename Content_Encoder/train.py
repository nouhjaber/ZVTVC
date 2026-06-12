import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchaudio
import numpy as np
from pathlib import Path
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

import sys
zvtvc_root = Path(__file__).parent.parent
if str(zvtvc_root) not in sys.path:
    sys.path.insert(0, str(zvtvc_root))

from Shard_dataset_unified import ContentEncoderDataset as ShardDataset

# CUDA optimizations
torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
except AttributeError:
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True

from model.encoder import CausalConv1d
from model.multi_scale_backbone import MultiScaleEncoder
from model.fusion import HierarchicalFusion
from model.bottleneck import InformationBottleneck
from model.phoneme_classifier import PhonemeClassifier
from model.speaker_adversarial import SpeakerAdversarial

from training.losses import MultiTaskLoss, LossScheduler
from training.trainer import Trainer
from training.distillation import DistillationLoss
from training.contrastive import ContrastiveLoss
from training.consistency import ConsistencyLoss

from teachers.teacher_manager import TeacherManager
from teachers.ema_teacher import EMATeacher


# =============================================================================
# GLOBAL MEL TRANSFORM (cached, GPU-accelerated)
# =============================================================================

_MEL_TRANSFORM_CACHE = {}

def get_mel_transform(device='cuda'):
    """Get cached mel transform for device."""
    if device not in _MEL_TRANSFORM_CACHE:
        _MEL_TRANSFORM_CACHE[device] = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=320,
            n_mels=80,
            f_min=0,
            f_max=8000,
        ).to(device)
    return _MEL_TRANSFORM_CACHE[device]


# =============================================================================
# BATCH ADAPTER (GPU mel computation + phoneme alignment)
# =============================================================================

def adapt_batch_for_content_encoder(batch, device='cuda'):
    """
    Convert ShardDataset batch to Content Encoder trainer format.
    Computes mel spectrogram from audio ON GPU.
    Aligns phoneme labels to mel time dimension.

    ShardDataset returns:     Content Encoder expects:
    - audio               →   - mel (computed on GPU)
    - phoneme_labels      →   - phoneme_labels (aligned to mel T)
    - phoneme_confidence  →   - phoneme_confidence (aligned to mel T)
    - speaker_id          →   - speaker_id
    """
    adapted = {}

    if 'mel' in batch:
        adapted['mel'] = batch['mel']
    elif 'audio' in batch:
        audio = batch['audio']
        if not audio.is_cuda and device != 'cpu':
            audio = audio.to(device, non_blocking=True)
        mel_transform = get_mel_transform(device)
        mel = mel_transform(audio)
        mel = torch.log(mel.clamp(min=1e-5))
        adapted['mel'] = mel
        adapted['audio'] = audio

    if 'audio' in batch and 'audio' not in adapted:
        adapted['audio'] = batch['audio']

    mel_T = adapted['mel'].shape[2] if 'mel' in adapted else None

    if 'phoneme_labels' in batch and mel_T is not None:
        ph = batch['phoneme_labels']
        if ph.shape[1] != mel_T:
            ph_aligned = torch.nn.functional.interpolate(
                ph.unsqueeze(1).float(), size=mel_T, mode='nearest'
            ).squeeze(1).long()
            adapted['phoneme_labels'] = ph_aligned
        else:
            adapted['phoneme_labels'] = ph
    elif 'phoneme_labels' in batch:
        adapted['phoneme_labels'] = batch['phoneme_labels']

    if 'phoneme_confidence' in batch and mel_T is not None:
        conf = batch['phoneme_confidence']
        if conf.shape[1] != mel_T:
            conf_aligned = torch.nn.functional.interpolate(
                conf.unsqueeze(1), size=mel_T, mode='nearest'
            ).squeeze(1)
            adapted['phoneme_confidence'] = conf_aligned
        else:
            adapted['phoneme_confidence'] = conf
    elif 'phoneme_confidence' in batch:
        adapted['phoneme_confidence'] = batch['phoneme_confidence']

    if 'speaker_id' in batch:
        adapted['speaker_id'] = batch['speaker_id']
    elif 'speaker_ids' in batch:
        adapted['speaker_id'] = batch['speaker_ids']

    return adapted


# =============================================================================
# AUTO-CALCULATION FUNCTIONS
# =============================================================================

def calculate_learning_rate(base_lr: float, batch_size: int, base_batch_size: int = 32) -> float:
    """
    Scale learning rate linearly with batch size.
    Formula: new_lr = base_lr * (batch_size / base_batch_size)
    Example: base_lr=1e-4, batch=64, base=32 → lr=2e-4
    """
    scale_factor = batch_size / base_batch_size
    scaled_lr = base_lr * scale_factor
    print(f"[LR] {base_lr:.6f} × {scale_factor:.2f} = {scaled_lr:.6f}  (batch {base_batch_size}→{batch_size})")
    return scaled_lr


def calculate_total_iterations(
    dataset_size: int,
    reference_dataset_hours: float = 400.0,
    reference_iterations: int = 225000
) -> Dict[str, int]:
    """
    Scale total iterations proportionally to dataset size.
    Formula: total = reference_iterations * (dataset_hours / reference_hours)
    Splits across stages: 22% / 33% / 12% / 33%
      Stage 0 (Bootstrap):           22%
      Stage 1 (Self-Shifting):       33%
      Stage 2 (Bottleneck Adapt):    12% — short, just stabilize at α=0.2
      Stage 3 (Full Adversarial):    33% — fresh adversarial warmup in stable space
    """
    dataset_hours = (dataset_size * 6.0) / 3600.0  # assume 6s avg per sample
    scale_factor = dataset_hours / reference_dataset_hours
    total = int(reference_iterations * scale_factor)

    stage_0 = int(total * 0.22)
    stage_1 = int(total * 0.33)
    stage_2 = int(total * 0.12)
    stage_3 = int(total * 0.33)

    print(f"[Iterations] Dataset={dataset_hours:.1f}h → total={total:,}  "
          f"(stage0={stage_0:,} / stage1={stage_1:,} / stage2={stage_2:,} / stage3={stage_3:,})")

    return {'total': total, 'stage_0': stage_0, 'stage_1': stage_1, 'stage_2': stage_2, 'stage_3': stage_3}


def calculate_warmup_iterations(total_iterations: int, warmup_ratio: float = 0.01) -> int:
    """Warmup = 1% of total iterations, minimum 500."""
    warmup = max(500, int(total_iterations * warmup_ratio))
    print(f"[Warmup] {warmup:,} iterations ({warmup_ratio*100:.0f}% of {total_iterations:,})")
    return warmup


# =============================================================================
# CONTENT ENCODER MODEL
# =============================================================================

class ContentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocess_conv = CausalConv1d(in_channels=80, out_channels=256, kernel_size=3)
        self.preprocess_activation = nn.LeakyReLU(negative_slope=0.2)
        self.preprocess_norm = nn.LayerNorm(256)
        self.backbone = MultiScaleEncoder(channels=256)
        self.fusion = HierarchicalFusion()
        self.bottleneck = InformationBottleneck(alpha_bn=0.5)
        self.output_proj = nn.Conv1d(256, 512, kernel_size=1)
        self.output_norm = nn.LayerNorm(512)

    def forward(self, mel_spec):
        x = self.preprocess_conv(mel_spec)
        x = self.preprocess_activation(x)
        x = x.transpose(1, 2)
        x = self.preprocess_norm(x)
        x = x.transpose(1, 2)
        fine, medium, coarse = self.backbone(x)
        fused = self.fusion(fine, medium, coarse)
        bottleneck_out = self.bottleneck(fused)
        z_c = self.output_proj(bottleneck_out)
        z_c = z_c.transpose(1, 2)
        z_c = self.output_norm(z_c)
        z_c = z_c.transpose(1, 2)
        return z_c


# =============================================================================
# DETAILED TRAINING LOGGER
# =============================================================================

class DetailedTrainingLogger:
    """Shows detailed metrics during training."""

    def __init__(self, stage: int, max_iterations: int, log_frequency: int = 100):
        self.stage = stage
        self.max_iterations = max_iterations
        self.log_frequency = log_frequency
        self.start_time = time.time()
        self.iteration_times = []
        self.last_log_time = time.time()
        self.loss_history = []
        self.stage_names = {0: "Bootstrap", 1: "Self-Shifting", 2: "Bottleneck Adapt", 3: "Full Adversarial"}

    def log(self, iteration: int, losses: Dict, stage_start: int = 0):
        current_time = time.time()
        iter_time = current_time - self.last_log_time
        self.last_log_time = current_time

        per_iter_time = iter_time / max(self.log_frequency, 1)
        self.iteration_times.append(per_iter_time)
        if len(self.iteration_times) > 100:
            self.iteration_times = self.iteration_times[-100:]
        avg_iter_time = sum(self.iteration_times) / len(self.iteration_times)

        stage_progress = iteration - stage_start
        progress_pct = (stage_progress / self.max_iterations) * 100
        remaining = self.max_iterations - stage_progress
        eta_str = str(timedelta(seconds=int(remaining * avg_iter_time)))
        elapsed_str = str(timedelta(seconds=int(current_time - self.start_time)))

        def get_val(key, default=0.0):
            v = losses.get(key, default)
            return v.item() if isinstance(v, torch.Tensor) else v

        total_loss = get_val('total')
        self.loss_history.append(total_loss)

        grad_norm    = get_val('encoder_grad_norm')
        output_mean  = get_val('output_mean')
        output_std   = get_val('output_std')
        phoneme_acc  = get_val('phoneme_accuracy')
        has_nan      = get_val('has_nan', False)

        loss_status = "✓" if 0.0 <= total_loss <= 7.0 else "⚠"
        grad_status = "✓" if 0.1 <= grad_norm <= 10.0 else ("↓VANISH" if grad_norm < 0.1 else "↑EXPLODE")
        mean_status = "✓" if abs(output_mean) <= 0.5 else "⚠"
        std_status  = "✓" if 0.3 <= output_std <= 0.8 else ("⚠COLLAPSE" if output_std < 0.3 else "⚠HIGH")
        acc_status  = "✓" if phoneme_acc >= 0.20 else "LOW"
        nan_status  = "🔴 NaN!" if has_nan else "✓"

        if len(self.loss_history) >= 10:
            recent_avg = sum(self.loss_history[-10:]) / 10
            older_avg  = sum(self.loss_history[-20:-10]) / 10 if len(self.loss_history) >= 20 else recent_avg
            trend = "↓" if recent_avg < older_avg else ("↑" if recent_avg > older_avg else "→")
        else:
            trend = "→"

        print(f"\n{'='*80}")
        print(f"STAGE {self.stage}: {self.stage_names.get(self.stage, '?')} | "
              f"Iter {iteration:,} ({progress_pct:.1f}%) | ETA: {eta_str}")
        print(f"{'='*80}")
        print(f"\n📊 CRITICAL METRICS:")
        print(f"   Loss:          {total_loss:.4f} {trend} [{loss_status}] (target: ~0.5-2.0)")
        print(f"   Gradient:      {grad_norm:.4f}   [{grad_status}] (target: 0.1-10)")
        print(f"   Output Mean:   {output_mean:.4f}   [{mean_status}] (target: ±0.5)")
        print(f"   Output Std:    {output_std:.4f}   [{std_status}] (target: 0.3-0.8)")
        print(f"   Phoneme Acc:   {phoneme_acc*100:.1f}%    [{acc_status}] (target: >20%)")
        print(f"   NaN Check:     {nan_status}")
        print(f"\n📉 LOSS BREAKDOWN:")
        print(f"   Total:         {total_loss:.6f}")
        print(f"   Distillation:  {get_val('distill'):.6f}")
        print(f"   Phoneme:       {get_val('phoneme'):.6f}")
        print(f"   Contrastive:   {get_val('contrast'):.6f}")
        print(f"   Consistency:   {get_val('consist'):.6f}")
        print(f"   Speaker Adv:   {get_val('speaker'):.6f}")
        print(f"\n⏱️  TIME: Elapsed {elapsed_str} | {1.0/max(avg_iter_time, 0.001):.1f} iter/s")
        print(f"{'='*80}\n")


# =============================================================================
# TRAINER
# =============================================================================

class FixedTrainer(Trainer):
    """Trainer with fixed iteration tracking, detailed logging, and AMP support."""

    def __init__(self, *args, use_amp=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._iteration = getattr(self, 'global_iteration', 0)
        self.stage_start_iteration = 0
        self._loaded_from_stage = -1

        self.best_val_metric = float('inf')
        self.best_ckpt_path = None

        self.use_amp = use_amp and self.device.startswith('cuda')
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print(f"[OK] AMP (Mixed Precision) ENABLED")

        max_iter = self._get_stage_iterations()
        log_freq = self.config.get('log_frequency', 100)
        self.detailed_logger = DetailedTrainingLogger(self.stage, max_iter, log_frequency=log_freq)

    def _get_stage_iterations(self):
        if 'iterations_per_stage' in self.config:
            return self.config['iterations_per_stage'].get(self.stage, 50000)
        elif 'max_iterations' in self.config:
            if isinstance(self.config['max_iterations'], dict):
                return self.config['max_iterations'].get(self.stage, 50000)
            return self.config['max_iterations']
        return 50000

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, value):
        self._iteration = value
        if hasattr(self, 'global_iteration'):
            object.__setattr__(self, 'global_iteration', value)

    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self._loaded_from_stage = checkpoint.get('stage', -1)
        self._iteration = checkpoint.get('iteration', checkpoint.get('global_iteration', 0))
        self.stage_start_iteration = checkpoint.get('stage_start_iteration', self._iteration)

        # --- Encoder & phoneme classifier: strict load (these must match exactly) ---
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.phoneme_classifier.load_state_dict(checkpoint['phoneme_classifier'])

        # --- Speaker adversarial: handle num_speakers mismatch ---
        # Between Colab sessions, shard read errors can cause a speaker to vanish
        # (if their only samples are in an unreadable shard region), changing the
        # speaker count. The checkpoint fc2 layer may have a different output size.
        # We resize fc2 to match the checkpoint since the adversarial head's exact
        # output dimension doesn't affect encoder quality.
        try:
            self.speaker_adversarial.load_state_dict(checkpoint['speaker_adversarial'])
        except RuntimeError as e:
            if "size mismatch" in str(e) and "fc2" in str(e):
                ckpt_sd = checkpoint['speaker_adversarial']
                ckpt_num_spk = ckpt_sd['fc2.weight'].shape[0]
                cur_num_spk = self.speaker_adversarial.num_speakers
                print(f"  [FIX] Speaker count mismatch: checkpoint={ckpt_num_spk}, current={cur_num_spk}")
                print(f"  [FIX] Resizing speaker_adversarial.fc2 to match checkpoint ({ckpt_num_spk} speakers)")
                # Resize the current model's fc2 to match checkpoint
                self.speaker_adversarial.fc2 = nn.Linear(
                    self.speaker_adversarial.hidden_dim, ckpt_num_spk
                ).to(self.device)
                self.speaker_adversarial.num_speakers = ckpt_num_spk
                self.speaker_adversarial.load_state_dict(checkpoint['speaker_adversarial'])
                # Also update multi_task_loss for consistency
                if hasattr(self.multi_task_loss, 'num_speakers'):
                    self.multi_task_loss.num_speakers = ckpt_num_spk
            else:
                raise

        # --- Multi-task loss: handle EMA teacher key mismatches ---
        # The EMA teacher may have been registered as a submodule of DistillationLoss
        # after the checkpoint was saved, adding ~118 new keys that the old checkpoint
        # doesn't have. We load with strict=False and re-sync the EMA teacher after.
        try:
            self.multi_task_loss.load_state_dict(checkpoint['multi_task_loss'])
        except RuntimeError as e:
            if "Missing key" in str(e) or "Unexpected key" in str(e):
                missing, unexpected = self.multi_task_loss.load_state_dict(
                    checkpoint['multi_task_loss'], strict=False
                )
                ema_missing = [k for k in (missing or []) if 'ema_teacher' in k]
                non_ema_missing = [k for k in (missing or []) if 'ema_teacher' not in k]
                if ema_missing:
                    print(f"  [OK] Skipped {len(ema_missing)} EMA teacher keys (will re-sync from encoder)")
                if non_ema_missing:
                    print(f"  [WARNING] Missing non-EMA keys: {non_ema_missing}")
                if unexpected:
                    print(f"  [WARNING] Unexpected keys: {unexpected}")
            else:
                raise

        # --- Optimizers: load with error tolerance ---
        # Optimizer state may fail to load if model architecture changed (e.g. fc2 resize)
        if 'encoder_optimizer' in checkpoint:
            try:
                self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            except Exception as e:
                print(f"  [WARNING] Encoder optimizer load failed (will reinit): {e}")
        if 'heads_optimizer' in checkpoint:
            try:
                self.heads_optimizer.load_state_dict(checkpoint['heads_optimizer'])
            except Exception as e:
                print(f"  [WARNING] Heads optimizer load failed (will reinit): {e}")
        if 'loss_optimizer' in checkpoint:
            try:
                self.loss_optimizer.load_state_dict(checkpoint['loss_optimizer'])
            except Exception as e:
                print(f"  [WARNING] Loss optimizer load failed (will reinit): {e}")

        # --- EMA teacher: restore or re-sync ---
        if checkpoint.get('ema_teacher') is not None and hasattr(self, 'ema_teacher'):
            try:
                self.ema_teacher.encoder.load_state_dict(checkpoint['ema_teacher'])
                self.ema_teacher.enabled = True
                print(f"  EMA teacher restored from checkpoint")
            except Exception as e:
                print(f"  [WARNING] Failed to restore EMA teacher: {e}")
                self.ema_teacher.encoder.load_state_dict(self.encoder.state_dict())
                print(f"  [OK] EMA teacher re-synced from encoder")
        else:
            # No EMA in checkpoint — sync from the just-loaded encoder weights
            if hasattr(self, 'ema_teacher'):
                self.ema_teacher.encoder.load_state_dict(self.encoder.state_dict())
                print(f"  [OK] EMA teacher synced from encoder (not in checkpoint)")

        # Re-sync the EMA teacher reference inside distillation loss
        if (hasattr(self, 'multi_task_loss') and
                hasattr(self.multi_task_loss, 'distillation') and
                hasattr(self.multi_task_loss.distillation, 'ema_teacher') and
                hasattr(self, 'ema_teacher')):
            self.multi_task_loss.distillation.ema_teacher = self.ema_teacher
            print(f"  [OK] Distillation EMA teacher re-synced")

        print(f"  Loaded from stage {self._loaded_from_stage}, iteration {self._iteration}")

    def save_checkpoint(self, final=False, best=False, tag=None):
        """
        Policy:
          - best  → stage_X_best.pt  (no optimizers, lightweight)
          - final → stage_X_final.pt (full, with optimizers for resume)
          - tag   → stage_X_{tag}_iter_{N}.pt (e.g. 'nan')
        """
        if best:
            name = f"stage_{self.stage}_best.pt"
        elif final:
            name = f"stage_{self.stage}_final.pt"
        elif tag is not None:
            name = f"stage_{self.stage}_{tag}_iter_{self.iteration}.pt"
        else:
            return

        path = self.checkpoint_dir / name

        if best and path.exists():
            try:
                path.unlink()
            except Exception:
                pass

        checkpoint = {
            'iteration': self.iteration,
            'global_iteration': self.iteration,
            'stage_iteration': self.iteration - self.stage_start_iteration,
            'stage': self.stage,
            'stage_start_iteration': self.stage_start_iteration,
            'encoder': self.encoder.state_dict(),
            'phoneme_classifier': self.phoneme_classifier.state_dict(),
            'speaker_adversarial': self.speaker_adversarial.state_dict(),
            'multi_task_loss': self.multi_task_loss.state_dict(),
            'config': self.config
        }

        if final or tag is not None:
            checkpoint['encoder_optimizer'] = self.encoder_optimizer.state_dict()
            checkpoint['heads_optimizer'] = self.heads_optimizer.state_dict()
            checkpoint['loss_optimizer'] = self.loss_optimizer.state_dict()
            checkpoint['ema_teacher'] = (
                self.ema_teacher.encoder.state_dict()
                if hasattr(self, 'ema_teacher') and self.ema_teacher.enabled else None
            )

        torch.save(checkpoint, path)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved: {path} ({size_mb:.0f} MB)")

    def validate(self):
        """Run validation and return metrics."""
        self.encoder.eval()
        self.phoneme_classifier.eval()
        self.speaker_adversarial.eval()

        if hasattr(self, 'val_loader'):
            val_loader = self.val_loader
        else:
            num_workers = int(self.config.get('num_workers', 0))
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=False,
                num_workers=num_workers,
                collate_fn=ShardDataset.collate_fn,
                pin_memory=self.device.startswith('cuda'),
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None
            )

        total_phoneme_correct = 0
        total_phoneme_samples = 0
        total_speaker_correct = 0
        total_speaker_samples = 0
        total_val_metric = 0.0
        total_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                batch = adapt_batch_for_content_encoder(batch, device=self.device)

                mel = batch['mel']
                z_c = self.encoder(mel)

                phoneme_logits = self.phoneme_classifier(z_c)
                phoneme_preds = torch.argmax(phoneme_logits, dim=1)
                phoneme_labels = batch['phoneme_labels']
                confidence = batch.get('phoneme_confidence', torch.ones_like(phoneme_labels, dtype=torch.float))

                correct = (phoneme_preds == phoneme_labels).float() * confidence
                total_phoneme_correct += correct.sum().item()
                total_phoneme_samples += confidence.sum().item()

                speaker_logits = self.speaker_adversarial(z_c)

                mel_positive = batch.get('mel_positive', mel)
                z_c_positive = self.encoder(mel_positive)

                val_losses = self.multi_task_loss(
                    z_c=z_c,
                    phoneme_logits=phoneme_logits,
                    speaker_logits=speaker_logits,
                    phoneme_labels=batch['phoneme_labels'],
                    phoneme_confidence=batch['phoneme_confidence'],
                    speaker_labels=batch['speaker_id'],
                    mel_spec=mel,
                    z_c_positive=z_c_positive,
                    audio_or_mel=mel,
                    encoder=self.encoder,
                    iteration=self.iteration,
                    stage=self.stage
                )
                metric = val_losses.get('distill', val_losses.get('total'))
                if isinstance(metric, torch.Tensor):
                    metric = metric.detach().float().item()
                total_val_metric += float(metric)
                total_val_batches += 1

                speaker_preds = torch.argmax(speaker_logits, dim=-1)
                speaker_labels = batch['speaker_id']
                total_speaker_correct += (speaker_preds == speaker_labels).sum().item()
                total_speaker_samples += speaker_labels.numel()

        self.encoder.train()
        self.phoneme_classifier.train()
        self.speaker_adversarial.train()

        phoneme_acc = total_phoneme_correct / max(total_phoneme_samples, 1)
        speaker_acc = total_speaker_correct / max(total_speaker_samples, 1)
        val_metric  = total_val_metric / max(total_val_batches, 1)

        print(f"\n📋 VALIDATION: Phoneme Acc: {phoneme_acc*100:.2f}% | "
              f"Speaker Acc: {speaker_acc*100:.2f}% | Val Metric: {val_metric:.6f}\n")

        return {'phoneme_accuracy': phoneme_acc, 'speaker_accuracy': speaker_acc, 'val_metric': val_metric}

    def train_iteration(self, batch: Dict) -> Dict:
        """Train one iteration with AMP support."""
        self.encoder.train()
        self.phoneme_classifier.train()
        self.speaker_adversarial.train()

        self.encoder_optimizer.zero_grad()
        self.heads_optimizer.zero_grad()
        self.loss_optimizer.zero_grad()

        stage_relative_iter = self.iteration - self.stage_start_iteration
        lambda_adv, lambda_grl = self.loss_scheduler.get_params(stage_relative_iter)
        self.multi_task_loss.update_adversarial_params(lambda_adv, lambda_grl)
        if hasattr(self.speaker_adversarial, 'set_lambda_grl'):
            self.speaker_adversarial.set_lambda_grl(lambda_grl)

        with autocast('cuda', enabled=self.use_amp):
            mel_spec = batch['mel']
            z_c = self.encoder(mel_spec)
            phoneme_logits = self.phoneme_classifier(z_c)
            speaker_logits = self.speaker_adversarial(z_c)

            mel_positive = batch.get('mel_positive', mel_spec)
            z_c_positive = self.encoder(mel_spec) if mel_positive is mel_spec else self.encoder(mel_positive)

            waveform = batch.get('audio', None)
            losses = self.multi_task_loss(
                z_c=z_c,
                phoneme_logits=phoneme_logits,
                speaker_logits=speaker_logits,
                phoneme_labels=batch['phoneme_labels'],
                phoneme_confidence=batch['phoneme_confidence'],
                speaker_labels=batch['speaker_id'],
                mel_spec=mel_spec,
                z_c_positive=z_c_positive,
                audio_or_mel=mel_spec,
                encoder=self.encoder,
                iteration=self.iteration,
                stage=self.stage,
                waveform=waveform
            )
            total_loss = losses['total']

        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.encoder_optimizer)
            self.scaler.unscale_(self.heads_optimizer)
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=float('inf'))
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(
                list(self.phoneme_classifier.parameters()) + list(self.speaker_adversarial.parameters()),
                max_norm=1.0
            )
            self.scaler.step(self.encoder_optimizer)
            self.scaler.step(self.heads_optimizer)
            self.scaler.step(self.loss_optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=float('inf'))
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(
                list(self.phoneme_classifier.parameters()) + list(self.speaker_adversarial.parameters()),
                max_norm=1.0
            )
            self.encoder_optimizer.step()
            self.heads_optimizer.step()
            self.loss_optimizer.step()

        warmup_iters = self.config.get('warmup_iterations', 3000)
        if self.iteration < warmup_iters:
            if hasattr(self, 'warmup_scheduler_encoder'):
                self.warmup_scheduler_encoder.step()
            if hasattr(self, 'warmup_scheduler_heads'):
                self.warmup_scheduler_heads.step()
        else:
            if hasattr(self, 'cosine_scheduler_encoder'):
                self.cosine_scheduler_encoder.step()
            if hasattr(self, 'cosine_scheduler_heads'):
                self.cosine_scheduler_heads.step()

        with torch.no_grad():
            losses['encoder_grad_norm'] = encoder_grad_norm
            losses['output_mean'] = z_c.mean()
            losses['output_std'] = z_c.std()
            losses['output_min'] = z_c.min()
            losses['output_max'] = z_c.max()
            losses['has_nan'] = torch.isnan(z_c).any() or torch.isnan(total_loss)

            phoneme_preds = torch.argmax(phoneme_logits, dim=1)
            correct = (phoneme_preds == batch['phoneme_labels']).float()
            conf = batch['phoneme_confidence']
            losses['phoneme_accuracy'] = (correct * conf).sum() / (conf.sum() + 1e-8)

        if hasattr(self, 'ema_teacher') and self.iteration >= self.config.get('ema', {}).get('start_iteration', 10000):
            if not self.ema_teacher.enabled:
                self.ema_teacher.enable()
            self.ema_teacher.update(self.encoder)

        return losses

    def train_stage(self, resume_from=None):
        """Train for one complete stage."""
        if resume_from:
            self.load_checkpoint(resume_from)

        stage_iterations = self._get_stage_iterations()

        if self._loaded_from_stage == self.stage:
            iterations_done = self.iteration - self.stage_start_iteration
            target_iteration = self.iteration + (stage_iterations - iterations_done)
        else:
            self.stage_start_iteration = self.iteration
            target_iteration = self.iteration + stage_iterations

        print(f"\n{'='*70}")
        print(f"STAGE {self.stage} TRAINING")
        print(f"{'='*70}")
        print(f"  Iterations for stage:  {stage_iterations:,}")
        print(f"  Current iteration:     {self.iteration:,}")
        print(f"  Target iteration:      {target_iteration:,}")
        print(f"  Remaining:             {target_iteration - self.iteration:,}")
        print(f"{'='*70}\n")

        if self.iteration >= target_iteration:
            print(f"Stage {self.stage} already complete!")
            return

        self._run_training_loop(target_iteration)
        self.save_checkpoint(final=True)

        print(f"\n{'='*70}")
        print(f"STAGE {self.stage} COMPLETE!")
        print(f"{'='*70}\n")

    def _run_training_loop(self, target_iteration):
        """Main training loop."""
        if hasattr(self, 'train_loader'):
            train_loader = self.train_loader
        else:
            num_workers = int(self.config.get('num_workers', 0))
            from Shard_dataset_unified import ShardAwareSampler
            sampler = ShardAwareSampler(self.train_dataset, shuffle=True)
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=ShardDataset.collate_fn,
                pin_memory=True,
                drop_last=True,
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else None
            )

        print("Starting training loop...\n")

        while self.iteration < target_iteration:
            for batch in train_loader:
                batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                batch = adapt_batch_for_content_encoder(batch, device=self.device)

                losses = self.train_iteration(batch)

                has_nan = losses.get('has_nan', False)
                if isinstance(has_nan, torch.Tensor):
                    has_nan = has_nan.item()
                if has_nan:
                    print("\n🔴 NaN DETECTED! Stopping training.")
                    self.save_checkpoint(tag='nan')
                    raise RuntimeError("NaN detected!")

                self.iteration += 1

                log_freq = self.config.get('log_frequency', 100)
                if self.iteration % log_freq == 0:
                    self.detailed_logger.log(self.iteration, losses, self.stage_start_iteration)

                val_freq = self.config.get('validation_frequency', 5000)
                if self.iteration % val_freq == 0:
                    val_out = self.validate()
                    metric = val_out.get('val_metric')
                    if metric is not None and metric < self.best_val_metric:
                        self.best_val_metric = metric
                        self.save_checkpoint(best=True)

                if self.iteration >= target_iteration:
                    break


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Content Encoder')

    # Data
    parser.add_argument('--shard_dir', type=str, default=None,
                        help='Directory with preprocessed shards (default: ../shards)')

    # Training
    parser.add_argument('--stage', type=int, default=0,
                        help='Stage to train (0, 1, or 2)')
    parser.add_argument('--train_all_stages', action='store_true',
                        help='Train all stages 0→1→2 sequentially')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Iterations per stage override (default: auto from dataset size)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume from')
    parser.add_argument('--validation_frequency', type=int, default=None,
                        help='Validate every N iterations (default: auto ~5000)')

    # Teachers
    parser.add_argument('--whisper_model', type=str, default='openai/whisper-large-v3')
    parser.add_argument('--mhubert_model', type=str, default='utter-project/mHuBERT-147')
    parser.add_argument('--use_small_models', action='store_true',
                        help='Use whisper-small + hubert-base (saves VRAM on T4)')

    args = parser.parse_args()

    # =================================================================
    # PRINT HEADER
    # =================================================================
    print("\n" + "#" * 80)
    print("# CONTENT ENCODER - TRAINING")
    print(f"# MODE: {'Multi-Stage (0→1→2→3)' if args.train_all_stages else f'Single Stage ({args.stage})'}")
    if not args.shard_dir:
        args.shard_dir = str(Path(__file__).parent.parent / "shards")
    print(f"# DATA: {args.shard_dir}")
    print("#" * 80 + "\n")

    # =================================================================
    # DATA LOADING
    # =================================================================
    print("=" * 80)
    print("LOADING DATA FROM SHARDS")
    print("=" * 80)

    if not Path(args.shard_dir).exists():
        raise FileNotFoundError(f"Shard directory not found: {args.shard_dir}")

    metadata_path = Path(args.shard_dir) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"  Shard metadata: {metadata.get('stats', {})}")

    train_dataset = ShardDataset(args.shard_dir, split='train', lang=None)
    val_dataset   = ShardDataset(args.shard_dir, split='val',   lang=None)

    if hasattr(train_dataset, 'num_speakers'):
        num_speakers = train_dataset.num_speakers
    else:
        # Fallback: scan a batch to count unique speakers
        temp_loader = DataLoader(
            train_dataset, batch_size=min(100, len(train_dataset)),
            shuffle=False, num_workers=0, collate_fn=ShardDataset.collate_fn
        )
        speaker_ids_set = set()
        for batch in temp_loader:
            ids = batch.get('speaker_id', batch.get('speaker_ids'))
            if ids is None:
                break
            speaker_ids_set.update(ids.tolist() if torch.is_tensor(ids) else ids)
            if len(speaker_ids_set) > 1000:
                break
        num_speakers = len(speaker_ids_set) if speaker_ids_set else 100
        del temp_loader

    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Speakers: {num_speakers}")

    # =================================================================
    # MODEL CREATION
    # =================================================================
    print("\n" + "-" * 60)
    print("INITIALIZING MODELS")
    print("-" * 60)

    encoder = ContentEncoder().to(args.device)
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")

    phoneme_classifier = PhonemeClassifier(
        input_dim=512, num_classes=78, hidden_dim=256
    ).to(args.device)

    speaker_adversarial = SpeakerAdversarial(
        input_dim=512, num_speakers=num_speakers, hidden_dim=256
    ).to(args.device)

    if args.use_small_models:
        args.whisper_model = 'openai/whisper-small'
        args.mhubert_model = 'facebook/hubert-base-ls960'

    print(f"\nLoading teachers: {args.whisper_model}, {args.mhubert_model}")
    teacher_manager = TeacherManager(
        student_encoder=encoder,
        whisper_model_name=args.whisper_model,
        mhubert_model_name=args.mhubert_model,
        device=args.device,
        auto_load=True
    )

    whisper_dim = teacher_manager.whisper.get_output_dim()
    mhubert_dim = teacher_manager.mhubert.get_output_dim()
    print(f"  Whisper dim: {whisper_dim}, mHuBERT dim: {mhubert_dim}")

    ema_teacher = EMATeacher(
        encoder,
        alpha=0.99 if args.stage == 0 else (0.995 if args.stage == 1 else 0.999),
        device=args.device
    )

    distillation_loss = DistillationLoss(
        student_dim=512, whisper_dim=whisper_dim, mhubert_dim=mhubert_dim,
    ).to(args.device)
    distillation_loss.load_teachers(
        whisper_model=teacher_manager.whisper,
        mhubert_model=teacher_manager.mhubert,
        ema_teacher=ema_teacher
    )

    contrastive_loss = ContrastiveLoss(temperature=0.07).to(args.device)
    consistency_loss = ConsistencyLoss().to(args.device)

    multi_task_loss = MultiTaskLoss(
        distillation_loss=distillation_loss,
        contrastive_loss=contrastive_loss,
        consistency_loss=consistency_loss,
        num_phoneme_classes=78,
        num_speakers=num_speakers
    ).to(args.device)

    loss_scheduler = LossScheduler()

    # =================================================================
    # DATALOADERS
    # =================================================================
    print("\n" + "=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)
    print(f"  batch_size={args.batch_size}, num_workers={args.num_workers}")

    from Shard_dataset_unified import ShardAwareSampler
    train_sampler = ShardAwareSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=ShardDataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ShardDataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    print(f"  Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # =================================================================
    # AUTO-CALCULATE TRAINING PARAMETERS
    # =================================================================
    print("\n" + "=" * 80)
    print("CALCULATING TRAINING PARAMETERS")
    print("=" * 80)

    # LR scales with batch size (linear scaling rule)
    encoder_lr = calculate_learning_rate(1e-4, args.batch_size, base_batch_size=32)
    heads_lr   = calculate_learning_rate(1e-3, args.batch_size, base_batch_size=32)

    # Iterations scale with dataset size (or use manual override)
    if args.iterations:
        stage_0_iters = args.iterations
        stage_1_iters = args.iterations
        stage_2_iters = args.iterations
        stage_3_iters = args.iterations
        total_iters   = args.iterations * 4
        print(f"[Iterations] Manual override: {args.iterations:,} per stage")
    else:
        iteration_plan = calculate_total_iterations(len(train_dataset))
        stage_0_iters = iteration_plan['stage_0']
        stage_1_iters = iteration_plan['stage_1']
        stage_2_iters = iteration_plan['stage_2']
        stage_3_iters = iteration_plan['stage_3']
        total_iters   = iteration_plan['total']

    warmup_iters = calculate_warmup_iterations(total_iters, warmup_ratio=0.01)

    print("=" * 80 + "\n")

    config = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'learning_rate': {'encoder': encoder_lr, 'heads': heads_lr},
        'iterations_per_stage': {0: stage_0_iters, 1: stage_1_iters, 2: stage_2_iters, 3: stage_3_iters},
        'warmup_iterations': warmup_iters,
        'checkpoint_frequency': args.validation_frequency or min(5000, max(500, stage_0_iters // 10)),
        'validation_frequency': args.validation_frequency or min(5000, max(500, stage_0_iters // 10)),
        'log_frequency': 100,
        'ema': {'alpha': {0: 0.99, 1: 0.995, 2: 0.999, 3: 0.999}, 'start_iteration': 10000}
    }

    # =================================================================
    # CREATE TRAINER
    # =================================================================
    print("-" * 60)
    print("INITIALIZING TRAINER")
    print("-" * 60)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    trainer = FixedTrainer(
        encoder=encoder,
        phoneme_classifier=phoneme_classifier,
        speaker_adversarial=speaker_adversarial,
        multi_task_loss=multi_task_loss,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        stage=args.stage,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        config=config
    )

    trainer.train_loader = train_loader
    trainer.val_loader   = val_loader

    # Sync EMA teachers so distillation loss and trainer share the same object
    if hasattr(trainer.multi_task_loss, 'distillation'):
        trainer.multi_task_loss.distillation.ema_teacher = trainer.ema_teacher
        print("[OK] EMA teacher synced")

    # =================================================================
    # TRAINING
    # =================================================================
    if args.train_all_stages:
        print("\n" + "#" * 80)
        print("# MULTI-STAGE TRAINING: 0 → 1 → 2 → 3")
        print("#" * 80)
        print(f"# batch={args.batch_size}  workers={args.num_workers}")
        print(f"# LR: encoder={encoder_lr:.6f}  heads={heads_lr:.6f}")
        print(f"# Iters: {stage_0_iters:,} + {stage_1_iters:,} + {stage_2_iters:,} + {stage_3_iters:,} = {total_iters:,}")
        print(f"# Warmup: {warmup_iters:,}")
        if torch.cuda.is_available():
            print(f"# GPU: {torch.cuda.get_device_name(0)}")
        print("#" * 80 + "\n")

        resume_path = args.resume
        start_stage = 0

        if resume_path:
            try:
                ckpt_header = torch.load(resume_path, map_location='cpu', weights_only=False)
                ckpt_stage  = ckpt_header.get('stage', 0)
                ckpt_iter   = ckpt_header.get('iteration', 0)
                del ckpt_header
                start_stage = ckpt_stage
                print(f"[RESUME] Stage {ckpt_stage}, iteration {ckpt_iter} → starting from stage {start_stage}")
            except Exception as e:
                print(f"[WARNING] Could not read checkpoint: {e} — starting from stage 0")
        else:
            for s in [3, 2, 1, 0]:
                final_path = os.path.join(args.checkpoint_dir, f"stage_{s}_final.pt")
                if os.path.exists(final_path):
                    if s < 3:
                        start_stage = s + 1
                        resume_path = final_path
                        print(f"[AUTO-RESUME] Found stage_{s}_final.pt → starting from stage {start_stage}")
                    else:
                        print(f"[AUTO-RESUME] All stages complete (stage_3_final.pt exists)")
                        start_stage = 4
                    break

        stage_names = {0: 'Bootstrap', 1: 'Self-Shifting', 2: 'Bottleneck Adapt', 3: 'Full Adversarial'}

        for stage in [0, 1, 2, 3]:
            if stage < start_stage:
                final_ckpt = os.path.join(args.checkpoint_dir, f"stage_{stage}_final.pt")
                if os.path.exists(final_ckpt):
                    resume_path = final_ckpt
                print(f"[SKIP] Stage {stage} already complete")
                continue

            print("\n" + "=" * 80)
            print(f"STAGE {stage}: {stage_names[stage]}")
            print(f"Iterations: {config['iterations_per_stage'][stage]:,}")
            print("=" * 80 + "\n")

            trainer.stage = stage
            trainer.config['iterations_per_stage'][stage] = config['iterations_per_stage'][stage]
            trainer.loss_scheduler.stage = stage

            ema_alpha = config['ema']['alpha'].get(stage, 0.999)
            if hasattr(trainer, 'ema_teacher'):
                trainer.ema_teacher.set_alpha(ema_alpha)
                if (hasattr(trainer.multi_task_loss, 'distillation') and
                        hasattr(trainer.multi_task_loss.distillation, 'ema_teacher') and
                        trainer.multi_task_loss.distillation.ema_teacher is not None):
                    trainer.multi_task_loss.distillation.ema_teacher.set_alpha(ema_alpha)

            # Bottleneck alpha schedule:
            #   Stage 0: α=0.5 (open, learn content freely)
            #   Stage 1: α=0.3 (moderate compression + adversarial warmup)
            #   Stage 2: α=0.2 (tighter compression, adversarial clamped while encoder adapts)
            #   Stage 3: α=0.2 (same compression, fresh adversarial warmup)
            bn_alpha = {0: 0.5, 1: 0.3, 2: 0.2, 3: 0.2}.get(stage, 0.2)
            if hasattr(trainer.encoder, 'bottleneck'):
                trainer.encoder.bottleneck.set_alpha_bn(bn_alpha)
                print(f"  Bottleneck alpha: {bn_alpha}")

            log_freq = config.get('log_frequency', 100)
            trainer.detailed_logger = DetailedTrainingLogger(
                stage, config['iterations_per_stage'][stage], log_frequency=log_freq
            )
            trainer.best_val_metric = float('inf')

            trainer.train_stage(resume_from=resume_path)
            resume_path = os.path.join(args.checkpoint_dir, f"stage_{stage}_final.pt")

            print(f"\n✓ Stage {stage} complete!")
            if stage < 3:
                print(f"  Resuming stage {stage + 1} from: {resume_path}\n")

        print("\n" + "#" * 80)
        print("# ALL STAGES COMPLETE! (0 → 1 → 2 → 3)")
        print("#" * 80 + "\n")

    else:
        print("\n" + "#" * 80)
        print(f"# STARTING TRAINING - STAGE {args.stage}")
        print("#" * 80)
        print(f"# batch={args.batch_size}  workers={args.num_workers}")
        print(f"# LR: encoder={encoder_lr:.6f}  heads={heads_lr:.6f}")
        print(f"# Iters: {config['iterations_per_stage'][args.stage]:,}")
        print(f"# Warmup: {warmup_iters:,}")
        if torch.cuda.is_available():
            print(f"# GPU: {torch.cuda.get_device_name(0)}")
        print("#" * 80 + "\n")

        trainer.train_stage(resume_from=args.resume)

        print("\n" + "#" * 80)
        print(f"# STAGE {args.stage} COMPLETE!")
        if args.stage < 2:
            print(f"# Next: python train.py --stage {args.stage + 1} "
                  f"--resume checkpoints/stage_{args.stage}_final.pt")
        print("#" * 80 + "\n")


if __name__ == '__main__':
    main()