import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import os
from pathlib import Path
import time
from datetime import datetime, timedelta

from training.losses import MultiTaskLoss, LossScheduler, MetricsTracker
from training.dataset import AudioDataset, collate_fn
from teachers.ema_teacher import EMATeacher, EMAScheduler


class TrainingLogger:
    """
    Comprehensive training logger with time estimates and progress tracking.
    """

    def __init__(self, stage: int, max_iterations: int, log_frequency: int = 100):
        self.stage = stage
        self.max_iterations = max_iterations
        self.log_frequency = log_frequency

        # Time tracking
        self.start_time = time.time()
        self.iteration_times = []
        self.last_log_time = time.time()

        # Progress tracking
        self.iterations_logged = 0

        # Stage names for better readability
        self.stage_names = {
            0: "Bootstrap",
            1: "Self-Shifting",
            2: "Full Training"
        }

    def log_iteration(
        self,
        iteration: int,
        stage_iteration: int,
        losses: Dict[str, torch.Tensor],
        metrics: Optional[Dict[str, float]] = None
    ):
        current_time = time.time()
        iter_time = current_time - self.last_log_time
        self.iteration_times.append(iter_time)
        self.last_log_time = current_time

        # Calculate averages over last 100 iterations
        if len(self.iteration_times) > 100:
            self.iteration_times = self.iteration_times[-100:]
        avg_iter_time = sum(self.iteration_times) / len(self.iteration_times)

        # Calculate progress within this stage
        progress_pct = (stage_iteration / max(self.max_iterations, 1)) * 100.0
        iterations_remaining = self.max_iterations - stage_iteration

        # Time estimates
        eta_seconds = max(iterations_remaining, 0) * avg_iter_time
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        # Total elapsed time
        total_elapsed = current_time - self.start_time
        elapsed_str = str(timedelta(seconds=int(total_elapsed)))

        # Print header every 20 logs
        if self.iterations_logged % 20 == 0:
            self._print_header()

        def _get_value(v, default=0.0):
            if v is None:
                return default
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
            return float(v)

        print(f"\n{'='*80}")
        print(
            f"STAGE {self.stage}: {self.stage_names.get(self.stage, 'Unknown')} | "
            f"Stage Iter {stage_iteration:,}/{self.max_iterations:,} ({progress_pct:.1f}%) | "
            f"Global Iter {iteration:,}"
        )
        print(f"{'='*80}")

        # Time information
        # FIXED: iter_time is for log_frequency iterations, not 1 iteration
        iter_time_per_iteration = iter_time / self.log_frequency
        iter_speed = 1.0 / max(iter_time_per_iteration, 1e-6)
        avg_time_per_iteration = avg_iter_time / self.log_frequency
        print("TIME:")
        print(f"   Iteration Time:     {iter_time_per_iteration:.3f}s ({iter_speed:.1f} iter/s)")
        print(f"   Avg Iter Time:      {avg_time_per_iteration:.3f}s (last 100 iters)")
        print(f"   Elapsed:            {elapsed_str}")
        print(f"   ETA This Stage:     {eta_str}")
        print(f"   Est. Completion:    {datetime.now() + timedelta(seconds=int(eta_seconds))}")

        # CRITICAL METRICS
        print(f"\n{'*'*80}")
        print("CRITICAL METRICS (CoLAP Analysis)")
        print(f"{'*'*80}")

        total_loss = _get_value(losses.get("total", 0.0))
        loss_status = "OK" if 4.0 <= total_loss <= 7.0 else "WARNING"
        print(f"   Loss Curve:         {total_loss:.4f} [Target: 6.0 -> 4.5] [{loss_status}]")

        encoder_grad = _get_value(losses.get("encoder_grad_norm", 0.0))
        grad_status = "OK" if 0.1 <= encoder_grad <= 10.0 else ("VANISHING" if encoder_grad < 0.1 else "EXPLODING")
        print(f"   Gradient Norm:      {encoder_grad:.4f} [Target: 0.1-10.0] [{grad_status}]")

        output_mean = _get_value(losses.get("output_mean", 0.0))
        mean_status = "OK" if abs(output_mean) <= 0.5 else "WARNING"
        print(f"   Output Mean:        {output_mean:.4f} [Target: ±0.5] [{mean_status}]")

        output_std = _get_value(losses.get("output_std", 0.0))
        std_status = "OK" if 0.3 <= output_std <= 0.8 else ("COLLAPSED" if output_std < 0.3 else "WARNING")
        print(f"   Output Std:         {output_std:.4f} [Target: 0.3-0.8] [{std_status}]")

        phoneme_acc = _get_value(losses.get("phoneme_accuracy", 0.0))
        acc_status = "OK" if phoneme_acc >= 0.20 else "LOW"
        print(f"   Phoneme Accuracy:   {phoneme_acc*100:.2f}% [Target: >20%] [{acc_status}]")

        has_nan = losses.get("has_nan", False)
        if isinstance(has_nan, torch.Tensor):
            has_nan = bool(has_nan.detach().cpu().item())
        nan_status = "ALERT! NaN DETECTED!" if has_nan else "OK"
        print(f"   NaN Check:          {'NaN FOUND' if has_nan else 'No NaN'} [{nan_status}]")

        print(f"{'*'*80}")

        # Loss information
        print("\nLOSSES:")
        print(f"   Total Loss:         {total_loss:.6f}")
        print(f"   Distillation:       {_get_value(losses.get('distill', 0.0)):.6f}")
        print(f"   Phoneme:            {_get_value(losses.get('phoneme', 0.0)):.6f}")
        print(f"   Speaker (Adv):      {_get_value(losses.get('speaker', 0.0)):.6f}")
        print(f"   Contrastive:        {_get_value(losses.get('contrast', 0.0)):.6f}")
        print(f"   Consistency:        {_get_value(losses.get('consist', 0.0)):.6f}")

        print(f"{'='*80}\n")
        self.iterations_logged += 1

    def _print_header(self):
        print(f"\n{'#'*80}")
        print(f"# TRAINING LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}\n")

    def log_stage_complete(self, total_time: float):
        print(f"\n{'#'*80}")
        print(f"# STAGE {self.stage} ({self.stage_names.get(self.stage, 'Unknown')}) COMPLETE!")
        print(f"# Total Time: {timedelta(seconds=int(total_time))}")
        print(f"# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}\n")


def deep_merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two config dictionaries.
    Override values take precedence, but nested dicts are merged recursively.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_configs(result[key], value)
        else:
            result[key] = value
    return result


class Trainer:
    """
    Main trainer for Content Encoder with 3-stage progressive training.

    FIXED:
      - No circular self-imports
      - Stage-relative iteration counting
      - Best + final checkpoint policy (optional intermediate)
      - validate() returns a loss-based metric for "best"
      - Fixed broken print strings and bad indentation/nesting
    """

    def __init__(
        self,
        encoder: nn.Module,
        phoneme_classifier: nn.Module,
        speaker_adversarial: nn.Module,
        multi_task_loss: MultiTaskLoss,
        train_dataset: AudioDataset,
        val_dataset: AudioDataset,
        stage: int = 0,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        config: Optional[Dict] = None
    ):
        self.device = device
        self.stage = stage

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Models
        self.encoder = encoder.to(device)
        self.phoneme_classifier = phoneme_classifier.to(device)
        self.speaker_adversarial = speaker_adversarial.to(device)

        # Loss
        self.multi_task_loss = multi_task_loss.to(device)

        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Config
        default_config = self._default_config()
        self.config = deep_merge_configs(default_config, config) if config is not None else default_config

        # Training state
        self.global_iteration = 0
        self.stage_iteration = 0
        self.epoch = 0

        # Optimizers + schedulers
        self._setup_optimizers()
        self.loss_scheduler = LossScheduler(stage)

        # EMA Teacher
        self.ema_teacher = EMATeacher(
            self.encoder,
            alpha=self.config["ema"]["alpha"][stage],
            device=device
        )
        if stage > 0 or self.global_iteration >= self.config["ema"]["start_iteration"]:
            self.ema_teacher.enable()

        # Metrics
        self.metrics_tracker = MetricsTracker()

        # Best checkpoint tracking
        self.best_val_metric = float("inf")
        self.best_checkpoint_path = None

        # Mode collapse detection
        self.mode_collapse_detector = ModeCollapseDetector()

        # Logger
        self.iterations_per_stage = int(self.config["iterations_per_stage"][self.stage])
        self.logger = TrainingLogger(
            stage=stage,
            max_iterations=self.iterations_per_stage,
            log_frequency=int(self.config["log_frequency"])
        )

        # Optional: bottleneck schedule if encoder exposes it
        self._update_bottleneck_alpha(stage)

    def _update_bottleneck_alpha(self, stage: int):
        alpha_schedule = {0: 0.5, 1: 0.3, 2: 0.1}
        alpha_bn = alpha_schedule.get(stage, 0.5)
        if hasattr(self.encoder, "bottleneck") and hasattr(self.encoder.bottleneck, "set_alpha_bn"):
            self.encoder.bottleneck.set_alpha_bn(alpha_bn)
            print(f"[OK] Bottleneck alpha_bn set to {alpha_bn} for stage {stage}")

    def _default_config(self) -> Dict:
        return {
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "learning_rate": {
                "encoder": 1e-4,
                "heads": 1e-3
            },
            "iterations_per_stage": {
                0: 50000,
                1: 75000,
                2: 100000
            },
            "checkpoint_frequency": 5000,
            "validation_frequency": 5000,
            "log_frequency": 100,
            # best+final only unless explicitly enabled
            "save_intermediate_checkpoints": False,
            "ema": {
                "alpha": {0: 0.99, 1: 0.995, 2: 0.999},
                "start_iteration": 10000
            }
        }

    def _setup_optimizers(self):
        self.encoder_optimizer = optim.AdamW(
            self.encoder.parameters(),
            lr=self.config["learning_rate"]["encoder"],
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        head_params = list(self.phoneme_classifier.parameters()) + list(self.speaker_adversarial.parameters())
        self.heads_optimizer = optim.AdamW(
            head_params,
            lr=self.config["learning_rate"]["heads"],
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Collect all trainable params from multi_task_loss (includes projection layers)
        # Filter out frozen params (like log_sigma which has requires_grad=False)
        loss_params = [p for p in self.multi_task_loss.parameters() if p.requires_grad]
        if not loss_params:
            # Fallback: use log_sigma as dummy so optimizer doesn't crash
            loss_params = [self.multi_task_loss.log_sigma]
        self.loss_optimizer = optim.AdamW(
            loss_params,
            lr=self.config["learning_rate"]["encoder"],
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

        warmup_iters = 3000

        def warmup_lambda(iteration):
            if iteration < warmup_iters:
                return iteration / float(warmup_iters)
            return 1.0

        self.warmup_scheduler_encoder = LambdaLR(self.encoder_optimizer, lr_lambda=warmup_lambda)
        self.warmup_scheduler_heads = LambdaLR(self.heads_optimizer, lr_lambda=warmup_lambda)

        self.cosine_scheduler_encoder = CosineAnnealingWarmRestarts(
            self.encoder_optimizer, T_0=30000, T_mult=2, eta_min=1e-6
        )
        self.cosine_scheduler_heads = CosineAnnealingWarmRestarts(
            self.heads_optimizer, T_0=30000, T_mult=2, eta_min=1e-6
        )

    def train_stage(self, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(self.config["batch_size"]),
            shuffle=True,
            num_workers=int(self.config.get("num_workers", 0)),
            collate_fn=collate_fn,
            pin_memory=self.device.startswith("cuda"),
            drop_last=True,
            persistent_workers=int(self.config.get("num_workers", 0)) > 0,
            prefetch_factor=2 if int(self.config.get("num_workers", 0)) > 0 else None,
        )

        target_stage_iterations = int(self.config["iterations_per_stage"][self.stage])

        print(f"\n{'='*60}")
        print(f"Starting Stage {self.stage} Training")
        print(f"{'='*60}")
        print(f"  Target iterations this stage: {target_stage_iterations:,}")
        print(f"  Starting stage iteration:     {self.stage_iteration:,}")
        print(f"  Global iteration:             {self.global_iteration:,}")
        print(f"  Remaining:                    {target_stage_iterations - self.stage_iteration:,}")
        print(f"{'='*60}\n")

        if self.stage_iteration >= target_stage_iterations:
            print(f"[INFO] Stage {self.stage} already completed ({self.stage_iteration}/{target_stage_iterations})")
            print("[INFO] To continue training, increment the stage number.")
            return

        stage_start_time = time.time()
        print("Loading first batch (this may take 30-60 seconds due to audio processing)...")
        first_batch = True

        while self.stage_iteration < target_stage_iterations:
            for batch in train_loader:
                if first_batch:
                    print("[OK] First batch loaded successfully!")
                    first_batch = False

                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                losses = self.train_iteration(batch)

                # NaN guard
                has_nan = losses.get("has_nan", False)
                if isinstance(has_nan, torch.Tensor):
                    has_nan = bool(has_nan.detach().cpu().item())
                if has_nan:
                    print("\n" + "!" * 80)
                    print("CRITICAL ERROR: NaN detected in outputs!")
                    print("Training halted to prevent further corruption.")
                    print("!" * 80)
                    # save emergency intermediate if allowed
                    self.save_checkpoint(final=False, best=False, force_intermediate=True)
                    raise RuntimeError("NaN detected in model outputs. Training stopped.")

                self.metrics_tracker.update(losses)

                if self.stage_iteration % int(self.config["log_frequency"]) == 0:
                    lambda_adv, lambda_grl = self.loss_scheduler.get_params(self.global_iteration)
                    losses["lambda_adv"] = lambda_adv
                    losses["lambda_grl"] = lambda_grl
                    self.logger.log_iteration(self.global_iteration, self.stage_iteration, losses)

                if self.stage_iteration % int(self.config["validation_frequency"]) == 0:
                    metrics = self.validate()
                    if metrics is not None:
                        val_metric = metrics.get("val_metric", None)
                        if val_metric is not None and val_metric < self.best_val_metric:
                            self.best_val_metric = float(val_metric)
                            self.save_checkpoint(best=True)

                if self.stage_iteration % int(self.config["checkpoint_frequency"]) == 0:
                    self.save_checkpoint()

                if self.stage_iteration % 1000 == 0 and self.stage_iteration > 0:
                    if self.mode_collapse_detector.check(losses):
                        print("\n⚠️  MODE COLLAPSE DETECTED! ⚠️")
                        self._handle_mode_collapse()

                self.stage_iteration += 1
                self.global_iteration += 1

                if self.stage_iteration >= target_stage_iterations:
                    break

        stage_elapsed = time.time() - stage_start_time
        print(f"\nStage {self.stage} completed!")
        self.logger.log_stage_complete(stage_elapsed)
        self.save_checkpoint(final=True)

    def train_iteration(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.encoder.train()
        self.phoneme_classifier.train()
        self.speaker_adversarial.train()

        grad_accum = int(self.config.get("gradient_accumulation_steps", 1))
        if self.stage_iteration % grad_accum == 0:
            self.encoder_optimizer.zero_grad()
            self.heads_optimizer.zero_grad()
            self.loss_optimizer.zero_grad()

        lambda_adv, lambda_grl = self.loss_scheduler.get_params(self.global_iteration)
        self.multi_task_loss.update_adversarial_params(lambda_adv, lambda_grl)
        if hasattr(self.speaker_adversarial, "set_lambda_grl"):
            self.speaker_adversarial.set_lambda_grl(lambda_grl)

        mel_spec = batch["mel"]
        z_c = self.encoder(mel_spec)

        phoneme_logits = self.phoneme_classifier(z_c)
        speaker_logits = self.speaker_adversarial(z_c)

        mel_spec_positive = batch.get("mel_positive", mel_spec)
        z_c_positive = self.encoder(mel_spec_positive)

        losses = self.multi_task_loss(
            z_c=z_c,
            phoneme_logits=phoneme_logits,
            speaker_logits=speaker_logits,
            phoneme_labels=batch["phoneme_labels"],
            phoneme_confidence=batch["phoneme_confidence"],
            speaker_labels=batch["speaker_id"],
            mel_spec=mel_spec,
            z_c_positive=z_c_positive,
            audio_or_mel=mel_spec,
            encoder=self.encoder,
            iteration=self.global_iteration,
            stage=self.stage
        )

        total_loss = losses["total"]
        (total_loss / grad_accum).backward()

        if (self.stage_iteration + 1) % grad_accum == 0:
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=float("inf"))
            heads_grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.phoneme_classifier.parameters()) + list(self.speaker_adversarial.parameters()),
                max_norm=float("inf")
            )

            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(
                list(self.phoneme_classifier.parameters()) + list(self.speaker_adversarial.parameters()),
                max_norm=1.0
            )

            self.encoder_optimizer.step()
            self.heads_optimizer.step()
            self.loss_optimizer.step()

            if self.global_iteration < 3000:
                self.warmup_scheduler_encoder.step()
                self.warmup_scheduler_heads.step()
            else:
                self.cosine_scheduler_encoder.step()
                self.cosine_scheduler_heads.step()

            losses["encoder_grad_norm"] = encoder_grad_norm
            losses["heads_grad_norm"] = heads_grad_norm

        with torch.no_grad():
            losses["output_mean"] = z_c.mean()
            losses["output_std"] = z_c.std()
            losses["output_min"] = z_c.min()
            losses["output_max"] = z_c.max()
            losses["has_nan"] = torch.isnan(z_c).any() or torch.isnan(total_loss).any()

            phoneme_preds = torch.argmax(phoneme_logits, dim=1)
            correct = (phoneme_preds == batch["phoneme_labels"]).float()
            correct = correct * batch["phoneme_confidence"]
            phoneme_acc = correct.sum() / (batch["phoneme_confidence"].sum() + 1e-8)
            losses["phoneme_accuracy"] = phoneme_acc

        if self.global_iteration >= int(self.config["ema"]["start_iteration"]):
            if not self.ema_teacher.enabled:
                self.ema_teacher.enable()
            self.ema_teacher.update(self.encoder)

        return losses

    def validate(self):
        self.encoder.eval()
        self.phoneme_classifier.eval()
        self.speaker_adversarial.eval()

        nw = int(self.config.get("num_workers", 0))
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=int(self.config.get("batch_size", 32)),
            shuffle=False,
            num_workers=nw,
            collate_fn=collate_fn,
            pin_memory=self.device.startswith("cuda"),
            persistent_workers=nw > 0,
            prefetch_factor=2 if nw > 0 else None,
        )

        total_phoneme_correct = 0.0
        total_phoneme_count = 0.0
        total_speaker_correct = 0.0
        total_speaker_count = 0.0
        total_val_metric = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                mel = batch["mel"]
                z_c = self.encoder(mel)

                phoneme_logits = self.phoneme_classifier(z_c)
                speaker_logits = self.speaker_adversarial(z_c)

                phoneme_preds = torch.argmax(phoneme_logits, dim=1)
                phoneme_labels = batch["phoneme_labels"]
                confidence = batch.get(
                    "phoneme_confidence",
                    torch.ones_like(phoneme_labels, dtype=torch.float, device=self.device)
                )
                correct = (phoneme_preds == phoneme_labels).float() * confidence
                total_phoneme_correct += correct.sum().item()
                total_phoneme_count += confidence.sum().item()

                speaker_preds = torch.argmax(speaker_logits, dim=1)
                speaker_labels = batch["speaker_id"]
                total_speaker_correct += (speaker_preds == speaker_labels).float().sum().item()
                total_speaker_count += float(speaker_labels.shape[0])

                mel_positive = batch.get("mel_positive", mel)
                z_c_positive = self.encoder(mel_positive)

                val_losses = self.multi_task_loss(
                    z_c=z_c,
                    phoneme_logits=phoneme_logits,
                    speaker_logits=speaker_logits,
                    phoneme_labels=batch["phoneme_labels"],
                    phoneme_confidence=batch["phoneme_confidence"],
                    speaker_labels=batch["speaker_id"],
                    mel_spec=mel,
                    z_c_positive=z_c_positive,
                    audio_or_mel=mel,
                    encoder=self.encoder,
                    iteration=self.global_iteration,
                    stage=self.stage
                )

                metric = val_losses.get("distill", val_losses.get("total", None))
                if metric is None:
                    metric_val = float("inf")
                elif torch.is_tensor(metric):
                    metric_val = float(metric.detach().cpu().item())
                else:
                    metric_val = float(metric)

                total_val_metric += metric_val
                total_batches += 1

        val_metric = total_val_metric / max(total_batches, 1)
        val_phoneme_acc = total_phoneme_correct / (total_phoneme_count + 1e-8)
        val_speaker_acc = total_speaker_correct / (total_speaker_count + 1e-8)

        print("\n" + "-" * 80)
        print(
            f"VALIDATION | stage={self.stage} | stage_iter={self.stage_iteration:,} | "
            f"global_iter={self.global_iteration:,}"
        )
        print(f"  val_metric (lower=better): {val_metric:.6f}")
        print(f"  phoneme_acc:              {val_phoneme_acc*100:.2f}%")
        print(f"  speaker_acc:              {val_speaker_acc*100:.2f}%")
        print("-" * 80 + "\n")

        self.encoder.train()
        self.phoneme_classifier.train()
        self.speaker_adversarial.train()

        return {
            "val_metric": val_metric,
            "val_phoneme_accuracy": val_phoneme_acc,
            "val_speaker_accuracy": val_speaker_acc,
        }

    def save_checkpoint(self, final: bool = False, best: bool = False, force_intermediate: bool = False):
        """
        Save checkpoint.

        Strategy:
          - stage_X_best.pt  : whenever validation metric improves
          - stage_X_final.pt : at the end of each stage
          - stage_X_iter_XXXXX.pt : only if save_intermediate_checkpoints=True (or force_intermediate=True)
        """
        save_intermediate = bool(self.config.get("save_intermediate_checkpoints", False)) or force_intermediate

        if best:
            checkpoint_name = f"stage_{self.stage}_best.pt"
        elif final:
            checkpoint_name = f"stage_{self.stage}_final.pt"
        else:
            if not save_intermediate:
                return
            checkpoint_name = f"stage_{self.stage}_iter_{self.stage_iteration}.pt"

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint = {
            "global_iteration": self.global_iteration,
            "stage_iteration": self.stage_iteration,
            "stage": self.stage,
            "encoder": self.encoder.state_dict(),
            "phoneme_classifier": self.phoneme_classifier.state_dict(),
            "speaker_adversarial": self.speaker_adversarial.state_dict(),
            "multi_task_loss": self.multi_task_loss.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "heads_optimizer": self.heads_optimizer.state_dict(),
            "loss_optimizer": self.loss_optimizer.state_dict(),
            "ema_teacher": self.ema_teacher.encoder.state_dict() if getattr(self.ema_teacher, "enabled", False) else None,
            "config": self.config,
            "best_val_metric": getattr(self, "best_val_metric", None),
        }

        torch.save(checkpoint, checkpoint_path)
        if best:
            self.best_checkpoint_path = str(checkpoint_path)

        tag = "BEST" if best else ("FINAL" if final else "ITER")
        print(f"✓ Checkpoint saved ({tag}): {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint with robust handling.
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.global_iteration = int(checkpoint.get("global_iteration", checkpoint.get("iteration", 0)))
        self.stage_iteration = int(checkpoint.get("stage_iteration", 0))
        self.stage = int(checkpoint.get("stage", self.stage))

        self.encoder.load_state_dict(checkpoint["encoder"])
        print("  ✓ Encoder loaded")

        self.phoneme_classifier.load_state_dict(checkpoint["phoneme_classifier"])
        print("  ✓ Phoneme classifier loaded")

        self.speaker_adversarial.load_state_dict(checkpoint["speaker_adversarial"])
        print("  ✓ Speaker adversarial loaded")

        try:
            self.multi_task_loss.load_state_dict(checkpoint["multi_task_loss"], strict=True)
            print("  ✓ Multi-task loss loaded")
        except Exception:
            self.multi_task_loss.load_state_dict(checkpoint["multi_task_loss"], strict=False)
            print("  ✓ Multi-task loss loaded (partial)")

        try:
            self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
            self.heads_optimizer.load_state_dict(checkpoint["heads_optimizer"])
            self.loss_optimizer.load_state_dict(checkpoint["loss_optimizer"])
            print("  ✓ Optimizers loaded")
        except Exception as e:
            print(f"  ⚠ Optimizer load failed (continuing): {e}")

        if checkpoint.get("ema_teacher") is not None and hasattr(self, "ema_teacher"):
            try:
                self.ema_teacher.encoder.load_state_dict(checkpoint["ema_teacher"])
                print("  ✓ EMA teacher loaded")
            except Exception as e:
                print(f"  ⚠ EMA teacher load failed: {e}")
                self.ema_teacher.encoder.load_state_dict(self.encoder.state_dict())
        else:
            if hasattr(self, "ema_teacher"):
                self.ema_teacher.encoder.load_state_dict(self.encoder.state_dict())

        if checkpoint.get("best_val_metric") is not None:
            self.best_val_metric = float(checkpoint["best_val_metric"])

        print("✓ Checkpoint loaded")
        print(f"  Global iteration: {self.global_iteration}")
        print(f"  Stage iteration:  {self.stage_iteration}")
        print(f"  Stage:            {self.stage}")

    def _handle_mode_collapse(self):
        """
        Handle mode collapse by rolling back and reducing adversarial strength.

        Note: if intermediate checkpoints are disabled, this will roll back to BEST (preferred),
        else to FINAL, else it will continue.
        """
        print("⚠ Recovering from mode collapse...")

        best_ckpt = self.checkpoint_dir / f"stage_{self.stage}_best.pt"
        final_ckpt = self.checkpoint_dir / f"stage_{self.stage}_final.pt"

        rollback_path = None
        if best_ckpt.exists():
            rollback_path = best_ckpt
        elif final_ckpt.exists():
            rollback_path = final_ckpt
        else:
            # try latest iter ckpt if they exist
            iters = sorted(self.checkpoint_dir.glob(f"stage_{self.stage}_iter_*.pt"))
            if len(iters) >= 2:
                rollback_path = iters[-2]
            elif len(iters) == 1:
                rollback_path = iters[-1]

        if rollback_path is None:
            print("   No checkpoint found to roll back to. Continuing with current state.")
            return

        print(f"   Rolling back to: {rollback_path}")
        self.load_checkpoint(str(rollback_path))

        # Reduce adversarial strength if schedule is list-like
        if hasattr(self.loss_scheduler, "schedule") and isinstance(self.loss_scheduler.schedule, list):
            new_schedule = []
            for item in self.loss_scheduler.schedule:
                try:
                    it, adv, grl = item
                    new_schedule.append((it, adv * 0.5, grl * 0.5))
                except Exception:
                    new_schedule.append(item)
            self.loss_scheduler.schedule = new_schedule
            print("   Reduced adversarial strength by 50%")


class ModeCollapseDetector:
    """Detect mode collapse during training."""

    def __init__(self, num_speakers: int = 100):
        self.history = {
            "speaker_accuracy": [],
            "z_c_variance": [],
            "phoneme_accuracy": [],
            "total_loss": []
        }
        self.num_speakers = num_speakers
        self.random_chance = 1.0 / max(num_speakers, 1)

    def add_metrics(self, speaker_accuracy: float, z_c_variance: float, phoneme_accuracy: float, total_loss: float):
        self.history["speaker_accuracy"].append(speaker_accuracy)
        self.history["z_c_variance"].append(z_c_variance)
        self.history["phoneme_accuracy"].append(phoneme_accuracy)
        self.history["total_loss"].append(total_loss)

    def check(self, losses: Optional[Dict] = None) -> bool:
        if losses is None:
            if len(self.history["speaker_accuracy"]) < 3:
                return False
            recent_metrics = {
                "speaker_accuracy": self.history["speaker_accuracy"][-1],
                "z_c_variance": self.history["z_c_variance"][-1],
                "phoneme_accuracy": self.history["phoneme_accuracy"][-1],
            }
        else:
            recent_metrics = {}
            if "output_std" in losses:
                val = losses["output_std"]
                recent_metrics["z_c_variance"] = float(val.item()) if isinstance(val, torch.Tensor) else float(val)
            if "phoneme_accuracy" in losses:
                val = losses["phoneme_accuracy"]
                recent_metrics["phoneme_accuracy"] = float(val.item()) if isinstance(val, torch.Tensor) else float(val)

        if recent_metrics.get("z_c_variance", 1.0) < 0.01:
            print(f"⚠️  Mode collapse sign: Z_c variance collapsed ({recent_metrics['z_c_variance']:.6f})")
            return True

        return False


def train_all_stages(
    encoder: nn.Module,
    phoneme_classifier: nn.Module,
    speaker_adversarial: nn.Module,
    multi_task_loss: MultiTaskLoss,
    train_dataset: AudioDataset,
    val_dataset: AudioDataset,
    device: str = "cuda",
    checkpoint_dir: str = "./checkpoints",
    config: Optional[Dict] = None,
    start_stage: int = 0
):
    """
    Train all stages sequentially.

    Args:
        start_stage: Stage to start from (0, 1, or 2)
    """
    for stage in range(start_stage, 3):
        print(f"\n{'#'*80}")
        print(f"# STARTING STAGE {stage}")
        print(f"{'#'*80}\n")

        trainer = Trainer(
            encoder=encoder,
            phoneme_classifier=phoneme_classifier,
            speaker_adversarial=speaker_adversarial,
            multi_task_loss=multi_task_loss,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            stage=stage,
            device=device,
            checkpoint_dir=checkpoint_dir,
            config=config
        )

        final_checkpoint = Path(checkpoint_dir) / f"stage_{stage}_final.pt"
        best_checkpoint = Path(checkpoint_dir) / f"stage_{stage}_best.pt"

        if stage > 0:
            prev_final = Path(checkpoint_dir) / f"stage_{stage-1}_final.pt"
            if prev_final.exists() and not final_checkpoint.exists():
                print(f"Loading previous stage checkpoint: {prev_final}")
                trainer.load_checkpoint(str(prev_final))
                trainer.stage_iteration = 0
                trainer.stage = stage

        trainer.train_stage()

        encoder = trainer.encoder
        phoneme_classifier = trainer.phoneme_classifier
        speaker_adversarial = trainer.speaker_adversarial
        multi_task_loss = trainer.multi_task_loss

    print("\n" + "=" * 80)
    print("ALL STAGES COMPLETE!")
    print("=" * 80)