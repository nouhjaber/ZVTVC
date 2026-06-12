"""
Generator Trainer
Complete training loop with checkpointing, validation, logging, and EMA
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from typing import Dict, Optional, List
from tqdm import tqdm
import logging

from .flow_matching import FlowMatchingLoss, ConditionalFlowMatching, ClassifierFreeGuidance, MultiResolutionSTFTLoss


class EMAModel:
    """Exponential Moving Average of model parameters"""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model: Model to track
            decay: EMA decay rate
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update EMA parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


class GeneratorTrainer:
    """
    Complete training pipeline for Generator (Module 4)
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        checkpoint_dir: str = './checkpoints',
        device: str = 'cuda',
        logger: Optional[logging.Logger] = None
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # Training state
        self.current_iteration = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.loss_history = []

        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_mixed_precision()
        self._setup_ema()

        self.logger.info(f"Trainer initialized. Device: {device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    def _setup_optimizer(self):
        """Setup optimizer"""
        opt_config = self.config.get('training', {}).get('optimizer', {})

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(opt_config.get('learning_rate', 1e-4)),
            betas=opt_config.get('betas', [0.9, 0.999]),
            weight_decay=float(opt_config.get('weight_decay', 0.01)),
            eps=float(opt_config.get('eps', 1e-8))
        )

        self.grad_clip_norm = float(opt_config.get('gradient_clip_norm', 1.0))

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        sched_config = self.config.get('training', {}).get('scheduler', {})
        total_steps = int(self.config.get('training', {}).get('total_iterations', 200000))

        self.warmup_steps = int(sched_config.get('warmup_steps', 5000))

        # Store the base LR before any warmup modifications
        self.base_lr = self.optimizer.param_groups[0]['lr']

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - self.warmup_steps, 1),
            eta_min=float(sched_config.get('min_lr', 1e-6))
        )

        # Track how many times we've called scheduler.step()
        # so we don't double-step on resume or at the warmup boundary
        self._scheduler_steps = 0

    def _setup_loss(self):
        """Setup loss function"""
        # Conditional Flow Matching
        cfm_config = self.config.get('flow_matching', {}).get('training', {})
        self.cfm = ConditionalFlowMatching(
            time_sampling=cfm_config.get('time_sampling', 'uniform')
        )

        # Flow matching loss
        loss_config = self.config.get('loss', {})
        self.criterion = FlowMatchingLoss(
            cfm=self.cfm,
            time_weighting=loss_config.get('time_weighting', {}).get('enabled', False)
        )

        # Multi-Resolution STFT Loss (auxiliary)
        stft_config = loss_config.get('auxiliary', {}).get('stft', {})
        if stft_config.get('enabled', False):
            self.stft_loss = MultiResolutionSTFTLoss(
                fft_sizes=stft_config.get('fft_sizes', [512, 1024, 2048]),
                sc_weight=stft_config.get('sc_weight', 1.0),
                mag_weight=stft_config.get('mag_weight', 1.0)
            ).to(self.device)
            self.stft_loss_weight = float(stft_config.get('weight', 0.1))
            self.use_stft_loss = True
            self.logger.info(f"Multi-Resolution STFT Loss enabled (weight={self.stft_loss_weight})")
        else:
            self.stft_loss = None
            self.stft_loss_weight = 0.0
            self.use_stft_loss = False

        # Classifier-free guidance (optional)
        cfg_config = self.config.get('flow_matching', {}).get('inference', {}).get('cfg', {})
        if cfg_config.get('enabled', False):
            self.cfg = ClassifierFreeGuidance(
                p_uncond=float(cfg_config.get('unconditional_prob', 0.1)),
                guidance_scale=float(cfg_config.get('guidance_scale', 1.5))
            )
        else:
            self.cfg = None

    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        mp_config = self.config.get('training', {}).get('mixed_precision', {})
        self.use_amp = mp_config.get('enabled', True)

        if self.use_amp:
            self.scaler = GradScaler()
            self.logger.info("Mixed precision (FP16) enabled")
        else:
            self.scaler = None

    def _setup_ema(self):
        """Setup exponential moving average"""
        ema_config = self.config.get('training', {}).get('ema', {})

        if ema_config.get('enabled', True):
            self.ema = EMAModel(
                self.model,
                decay=float(ema_config.get('decay', 0.9999))
            )
            self.use_ema = True
            self.logger.info("EMA enabled")
        else:
            self.ema = None
            self.use_ema = False

    def get_lr(self) -> float:
        """Get current effective learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def _update_lr(self):
        """Update learning rate: linear warmup then cosine decay."""
        if self.current_iteration < self.warmup_steps:
            # Linear warmup: scale base_lr by progress fraction
            # Start from 1/warmup_steps (not 0) so first step has nonzero LR
            warmup_factor = max(self.current_iteration, 1) / max(self.warmup_steps, 1)
            lr = self.base_lr * warmup_factor
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        else:
            # Cosine annealing phase
            # How many cosine steps should have happened by now
            expected_steps = self.current_iteration - self.warmup_steps
            # Only step if we haven't already (avoids double-stepping on resume)
            while self._scheduler_steps < expected_steps:
                self.scheduler.step()
                self._scheduler_steps += 1
            # At the exact transition (expected_steps=0), LR stays at base_lr
            # because we haven't called scheduler.step() yet
            if expected_steps == 0:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.base_lr

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Dictionary with mel, content, prosody, timbre

        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()

        # Move to device
        mel = batch['mel'].to(self.device)
        content = batch['content'].to(self.device)
        prosody = batch['prosody'].to(self.device)
        timbre = batch['timbre'].to(self.device)

        # Apply classifier-free guidance (drop conditions randomly)
        if self.cfg is not None:
            content, prosody, timbre = self.cfg.drop_conditions(
                content, prosody, timbre, training=True
            )

        # Forward pass with mixed precision
        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        with autocast(device_type, enabled=self.use_amp):
            loss_dict = self.criterion(
                model=self.model,
                mel_target=mel,
                content=content,
                prosody=prosody,
                timbre=timbre,
                return_components=True  # Get velocity prediction for analysis
            )
            flow_loss = loss_dict['loss']

            # MONITORING: Capture velocity prediction statistics
            if 'v_pred' in loss_dict:
                with torch.no_grad():
                    v_pred = loss_dict['v_pred'].detach().float()
                    v_std = v_pred.std().item()
                    v_mean = v_pred.mean().item()
                    v_min = v_pred.min().item()
                    v_max = v_pred.max().item()
            else:
                v_std = v_mean = v_min = v_max = 0.0

            # Compute STFT loss if enabled
            stft_loss_value = 0.0
            if self.use_stft_loss and 'x_t' in loss_dict:
                # Get predicted mel from velocity: mel_pred = x_t + v_pred
                # At t=1, x_1 = x_0 + v, so mel_pred approximates target
                # For auxiliary loss, we use x_t + (1-t)*v_pred as approximation
                t = loss_dict.get('t', torch.zeros(1, device=mel.device))
                x_t = loss_dict['x_t']
                v_pred_tensor = loss_dict['v_pred']
                
                # Approximate mel prediction: x_t + remaining_velocity
                # mel_pred ≈ x_t + (1-t) * v_pred
                t_expanded = t.view(-1, 1, 1)
                mel_pred = x_t + (1 - t_expanded) * v_pred_tensor
                
                # Compute STFT loss
                stft_dict = self.stft_loss(mel_pred, mel)
                stft_loss_value = stft_dict['stft_loss']

            # Combined loss
            loss = flow_loss + self.stft_loss_weight * stft_loss_value

        # MONITORING: Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.error(f"❌ NaN/Inf detected in loss at iteration {self.current_iteration}!")
            return {
                'loss': float('nan'),
                'lr': self.get_lr(),
                'nan_detected': True
            }

        # Backward pass
        self.optimizer.zero_grad()

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # MONITORING: Capture gradient norm BEFORE clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            grad_norm = grad_norm.item()

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()

            # MONITORING: Capture gradient norm BEFORE clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            grad_norm = grad_norm.item()

            self.optimizer.step()

        # Update EMA
        if self.use_ema:
            self.ema.update(self.model)

        # Update learning rate (warmup + cosine)
        self._update_lr()

        # MONITORING: Return comprehensive metrics
        metrics = {
            'loss': loss.item(),
            'flow_loss': flow_loss.item() if torch.is_tensor(flow_loss) else flow_loss,
            'stft_loss': stft_loss_value.item() if torch.is_tensor(stft_loss_value) else stft_loss_value,
            'lr': self.get_lr(),
            'grad_norm': grad_norm,
            'velocity_std': v_std,
            'velocity_mean': v_mean,
            'velocity_min': v_min,
            'velocity_max': v_max,
            'nan_detected': False
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validation pass with comprehensive monitoring.

        Returns:
            Dictionary with validation metrics
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()

        # Use EMA model if available
        if self.use_ema:
            self.ema.apply_shadow(self.model)

        total_loss = 0.0
        num_batches = 0
        mel_generated = None  # Store one for analysis

        for batch in self.val_dataloader:
            mel = batch['mel'].to(self.device)
            content = batch['content'].to(self.device)
            prosody = batch['prosody'].to(self.device)
            timbre = batch['timbre'].to(self.device)

            device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
            with autocast(device_type, enabled=self.use_amp):
                loss_dict = self.criterion(
                    model=self.model,
                    mel_target=mel,
                    content=content,
                    prosody=prosody,
                    timbre=timbre,
                    return_components=False
                )
                loss = loss_dict['loss']

            total_loss += loss.item()
            num_batches += 1

            # MONITORING: Test inference (generate mel via 5-step Euler ODE)
            if mel_generated is None and num_batches == 1:
                try:
                    B, C, T = mel.shape
                    x = torch.randn(1, C, T, device=self.device)
                    num_steps = 5
                    dt = 1.0 / num_steps

                    for step_i in range(num_steps):
                        t_val = step_i / num_steps
                        t_batch = torch.full((1,), t_val, device=self.device)
                        v_pred = self.model(
                            x, t_batch,
                            content[:1], prosody[:1], timbre[:1]
                        )
                        x = x + dt * v_pred

                    mel_generated = x[0].cpu()

                except Exception as e:
                    self.logger.warning(f"⚠️ Inference test failed: {e}")
                    mel_generated = torch.zeros(C, T)  # Dummy

        avg_loss = total_loss / max(num_batches, 1)

        # MONITORING: Analyze generated mel spectrogram
        if mel_generated is not None:
            mel_min = mel_generated.min().item()
            mel_max = mel_generated.max().item()
            mel_mean = mel_generated.mean().item()
            mel_std = mel_generated.std().item()

            # Check if mel has structure (variance across frequency bins)
            mel_freq_var = mel_generated.var(dim=0).mean().item()  # Variance per time step

            inference_ok = not (torch.isnan(mel_generated).any() or torch.isinf(mel_generated).any())
        else:
            mel_min = mel_max = mel_mean = mel_std = mel_freq_var = 0.0
            inference_ok = False

        # Restore original model
        if self.use_ema:
            self.ema.restore(self.model)

        return {
            'val_loss': avg_loss,
            'mel_min': mel_min,
            'mel_max': mel_max,
            'mel_mean': mel_mean,
            'mel_std': mel_std,
            'mel_freq_var': mel_freq_var,
            'inference_ok': inference_ok
        }

    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'iteration': self.current_iteration,
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scheduler_steps': self._scheduler_steps,
            'best_loss': self.best_loss,
            'loss_history': self.loss_history,
            'config': self.config
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if self.use_ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_iter_{self.current_iteration:06d}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

        # Keep only last N checkpoints
        keep_last_n = self.config.get('checkpointing', {}).get('keep_last_n', 5)
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_iter_*.pt'))
        if len(checkpoints) > keep_last_n:
            for old_ckpt in checkpoints[:-keep_last_n]:
                old_ckpt.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_iteration = checkpoint['iteration']
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.loss_history = checkpoint.get('loss_history', [])
        self._scheduler_steps = checkpoint.get('scheduler_steps', 0)

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.use_ema and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

        self.logger.info(f"Loaded checkpoint from iteration {self.current_iteration}")

    def train(self, total_iterations: Optional[int] = None):
        """
        Main training loop with comprehensive monitoring.
        """
        if total_iterations is None:
            total_iterations = self.config.get('training', {}).get('total_iterations', 200000)

        val_frequency = self.config.get('training', {}).get('validation', {}).get('frequency', 5000)
        save_frequency = self.config.get('checkpointing', {}).get('save_frequency', 5000)
        log_frequency = self.config.get('training', {}).get('logging', {}).get('log_frequency', 100)

        self.logger.info(f"Starting training for {total_iterations} iterations "
                         f"(resuming from {self.current_iteration})")

        start_time = time.time()
        iter_times = []  # rolling window for speed calc
        loss_window = []  # rolling window for trend

        while self.current_iteration < total_iterations:
            for batch in self.train_dataloader:
                if self.current_iteration >= total_iterations:
                    break

                iter_start = time.time()

                # Training step
                metrics = self.train_step(batch)

                iter_elapsed = time.time() - iter_start
                iter_times.append(iter_elapsed)
                if len(iter_times) > 100:
                    iter_times = iter_times[-100:]

                # Track loss for trend
                loss_val = metrics.get('loss', 0.0)
                if not (loss_val != loss_val):  # skip NaN
                    loss_window.append(loss_val)
                if len(loss_window) > 200:
                    loss_window = loss_window[-200:]

                # ============================================================
                # LOGGING (skip iteration 0 — no meaningful data yet)
                # ============================================================
                if self.current_iteration > 0 and self.current_iteration % log_frequency == 0:
                    avg_iter = sum(iter_times) / len(iter_times)
                    steps_per_sec = 1.0 / max(avg_iter, 1e-6)
                    remaining = total_iterations - self.current_iteration
                    eta_sec = remaining * avg_iter
                    eta_h = eta_sec / 3600
                    progress = self.current_iteration / total_iterations * 100
                    elapsed_h = (time.time() - start_time) / 3600

                    # Loss trend
                    if len(loss_window) >= 20:
                        recent = sum(loss_window[-10:]) / 10
                        older = sum(loss_window[-20:-10]) / 10
                        trend = "↓" if recent < older - 0.01 else ("↑" if recent > older + 0.01 else "→")
                    else:
                        trend = "→"

                    loss = metrics['loss']
                    flow_loss = metrics.get('flow_loss', loss)
                    grad_norm = metrics.get('grad_norm', 0.0)
                    v_std = metrics.get('velocity_std', 0.0)
                    v_mean = metrics.get('velocity_mean', 0.0)
                    v_min = metrics.get('velocity_min', 0.0)
                    v_max = metrics.get('velocity_max', 0.0)
                    nan_detected = metrics.get('nan_detected', False)

                    # Status indicators
                    loss_status = "✅" if 0.3 <= loss <= 1.0 else ("⚠️" if loss < 2.0 else "❌")
                    grad_status = "✅" if 0.01 <= grad_norm <= 1.0 else ("⚠️" if grad_norm <= 10.0 else "❌")
                    v_status = "✅" if v_std > 0.1 else "❌"
                    nan_status = "❌" if nan_detected else "✅"

                    print(f"\n{'='*80}")
                    print(f"ITER {self.current_iteration}/{total_iterations} "
                          f"({progress:.1f}%) | Epoch {self.current_epoch} | "
                          f"ETA: {eta_h:.2f}h | Elapsed: {elapsed_h:.2f}h")
                    print(f"{'='*80}")

                    print(f"\n📊 CRITICAL METRICS:")
                    print(f"   Loss:          {loss:.4f} {trend}  [{loss_status}] (target: ~1.0 start → 0.3-0.4)")
                    print(f"   Gradient:      {grad_norm:.4f}    [{grad_status}] (target: 0.01-1.0)")
                    print(f"   Velocity Std:  {v_std:.4f}    [{v_status}] (target: >0.1, not collapsed)")
                    print(f"   NaN Check:     {nan_status}")

                    print(f"\n📉 LOSS BREAKDOWN:")
                    print(f"   Flow Matching:  {flow_loss:.6f}")
                    if self.use_stft_loss:
                        stft_loss = metrics.get('stft_loss', 0.0)
                        print(f"   STFT Aux:       {stft_loss:.6f} (weight={self.stft_loss_weight})")

                    print(f"\n📈 VELOCITY PREDICTION:")
                    print(f"   mean={v_mean:.4f}, std={v_std:.4f}, "
                          f"range=[{v_min:.3f}, {v_max:.3f}]")

                    print(f"\n⏱️  SPEED: {steps_per_sec:.2f} it/s | "
                          f"LR: {metrics['lr']:.6f}")
                    print(f"{'='*80}")

                    self.loss_history.append({
                        'iteration': self.current_iteration,
                        'loss': loss,
                        'flow_loss': flow_loss,
                        'lr': metrics['lr'],
                        'grad_norm': grad_norm,
                        'velocity_std': v_std,
                    })

                # ============================================================
                # VALIDATION (skip iteration 0)
                # ============================================================
                if (self.current_iteration > 0
                        and self.current_iteration % val_frequency == 0
                        and self.val_dataloader is not None):
                    val_metrics = self.validate()

                    val_loss = val_metrics.get('val_loss', float('inf'))
                    mel_min = val_metrics.get('mel_min', 0.0)
                    mel_max = val_metrics.get('mel_max', 0.0)
                    mel_mean = val_metrics.get('mel_mean', 0.0)
                    mel_std = val_metrics.get('mel_std', 0.0)
                    mel_freq_var = val_metrics.get('mel_freq_var', 0.0)
                    inference_ok = val_metrics.get('inference_ok', False)

                    val_status = "✅" if val_loss < loss + 0.2 else "⚠️"
                    mel_range_ok = (-5.0 <= mel_min) and (mel_max <= 5.0)
                    mel_status = "✅" if mel_range_ok else "⚠️"
                    struct_status = "✅" if mel_freq_var > 0.01 else "❌"
                    inf_status = "✅" if inference_ok else "❌"

                    print(f"\n{'='*80}")
                    print(f"🔍 VALIDATION @ iter {self.current_iteration}")
                    print(f"{'='*80}")
                    print(f"   {val_status} Val Loss:        {val_loss:.4f}")
                    print(f"   {mel_status} Mel Range:       [{mel_min:.2f}, {mel_max:.2f}] "
                          f"(target: -5 to +5, normalized)")
                    print(f"      Mel stats:      mean={mel_mean:.3f}, std={mel_std:.3f}")
                    print(f"   {struct_status} Mel Structure:   freq_var={mel_freq_var:.4f} "
                          f"(target: >0.01)")
                    print(f"   {inf_status} Inference Test:  "
                          f"{'OK' if inference_ok else 'FAILED (NaN/Inf)'}")

                    # Best model check
                    is_best = val_loss < self.best_loss
                    if is_best:
                        self.best_loss = val_loss
                        print(f"   🏆 NEW BEST MODEL! (val_loss={val_loss:.4f})")
                        self.save_checkpoint(is_best=True)

                    print(f"{'='*80}")

                # ============================================================
                # SAVE CHECKPOINT (skip iteration 0)
                # ============================================================
                if self.current_iteration > 0 and self.current_iteration % save_frequency == 0:
                    self.save_checkpoint(is_best=False)

                self.current_iteration += 1

            self.current_epoch += 1

        # Final checkpoint
        self.save_checkpoint(is_best=False)
        total_time = (time.time() - start_time) / 3600
        self.logger.info(f"Training completed! Total time: {total_time:.2f}h")