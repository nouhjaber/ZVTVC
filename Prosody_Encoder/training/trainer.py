"""
Trainer for Prosody Encoder
Handles training loop, checkpointing, logging, and validation
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import time
import numpy as np

from model.prosody_encoder import ProsodyEncoder
from training.losses import ProsodyLoss


def _adapt_batch_if_needed(batch):
    """
    Adapt batch from ShardDataset to Prosody Encoder format.
    
    ShardDataset provides dict with: f0, energy, voicing, rhythm, speaker_ids
    Prosody Encoder expects: (features, features_aug) where features is [B, 4, T]
    """
    # If already a tuple (features, features_aug), return as is
    if isinstance(batch, tuple) and len(batch) == 2:
        return batch
    
    # If dict from ShardDataset, convert
    if isinstance(batch, dict) and 'f0' in batch:
        channels = [batch['f0'], batch['energy'], batch['voicing']]
        if 'rhythm' in batch:
            channels.append(batch['rhythm'])
        # Stack → [B, 3or4, T]
        features = torch.stack(channels, dim=1)
        
        # No augmentation from shards
        features_aug = None
        
        return features, features_aug
    
    # Fallback: return as is
    return batch


class ProsodyTrainer:
    """Trainer for Prosody Encoder"""

    def __init__(
        self,
        model: ProsodyEncoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[ProsodyLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        max_iterations: int = 20000,
        log_interval: int = 100,
        val_interval: int = 500,
        checkpoint_interval: int = 1000,
        keep_last_n: int = 5,
        grad_clip_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.max_iterations = max_iterations
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.checkpoint_interval = checkpoint_interval
        self.keep_last_n = keep_last_n
        self.grad_clip_norm = grad_clip_norm

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Loss function
        if loss_fn is None:
            self.loss_fn = ProsodyLoss()
        else:
            self.loss_fn = loss_fn

        # Optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=1e-3,
                weight_decay=1e-4,
                betas=(0.9, 0.999),
            )
        else:
            self.optimizer = optimizer

        # Scheduler
        if scheduler is None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_iterations,
                eta_min=1e-5,
            )
        else:
            self.scheduler = scheduler

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Logging
        self.train_losses = []
        self.val_losses = []

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.max_iterations} iterations...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        print("\n" + "="*120)
        print("MONITORING METRICS - Expected Healthy Ranges:")
        print("="*120)
        print("  Loss:      Should decrease steadily")
        print("  GradNorm:  0.1 to 10.0 (gradient flow health)")
        print("  F0_mean:   ≈0.0 (±0.5) - whitened F0 should be centered")
        print("  F0_std:    ≈1.0 (0.5-1.5) - whitened F0 should have unit variance")
        print("  F0_corr:   >0.5 (Pearson correlation input/output - reconstruction quality)")
        print("  Out_std:   >0.2 (output diversity - checks for mode collapse)")
        print("  NaN:       Should always be NO (training stability)")
        print("="*120)
        print("\nWarnings will be displayed if metrics fall outside healthy ranges.\n")

        self.model.train()
        train_iter = iter(self.train_loader)

        start_time = time.time()
        remaining = self.max_iterations - self.global_step
        if remaining <= 0:
            print(f"Already at {self.global_step}/{self.max_iterations} — nothing to do.")
            return

        print(f"Running {remaining:,} iterations (from {self.global_step:,} to {self.max_iterations:,})")

        for step in range(remaining):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Adapt batch if needed (for ShardDataset compatibility)
            features, features_aug = _adapt_batch_if_needed(batch)

            # Move to device
            features = features.to(self.device)
            if features_aug is not None:
                features_aug = features_aug.to(self.device)

            # Forward pass (clean)
            output, explicit, reconstructions = self.model(
                explicit_features=features,
                return_reconstructions=True
            )

            # Forward pass (augmented) for consistency loss
            output_aug = None
            if features_aug is not None:
                output_aug, _, _ = self.model(
                    explicit_features=features_aug,
                    return_reconstructions=False
                )

            # Compute loss
            losses = self.loss_fn(
                refined=output,
                reconstructions=reconstructions,
                targets=features,
                refined_augmented=output_aug,
            )

            # Backward pass
            self.optimizer.zero_grad()
            losses["total"].backward()

            # Compute gradient norm BEFORE clipping
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            self.global_step += 1

            # Logging
            if self.global_step % self.log_interval == 0:
                elapsed = time.time() - start_time
                iter_per_sec = self.global_step / elapsed
                eta_seconds = (self.max_iterations - self.global_step) / iter_per_sec

                # Compute diagnostic metrics
                with torch.no_grad():
                    # F0 input statistics (index 0 is F0)
                    f0_input = features[:, 0, :].cpu().numpy()
                    f0_mean = np.mean(f0_input)
                    f0_std = np.std(f0_input)

                    # F0 correlation (input vs reconstructed)
                    if 'f0' in reconstructions:
                        f0_recon = reconstructions['f0'].cpu().numpy()
                        # Flatten for correlation
                        f0_input_flat = f0_input.flatten()
                        f0_recon_flat = f0_recon.flatten()
                        # Guard: corrcoef returns NaN for constant arrays
                        if np.std(f0_input_flat) > 1e-8 and np.std(f0_recon_flat) > 1e-8:
                            f0_corr = np.corrcoef(f0_input_flat, f0_recon_flat)[0, 1]
                            if np.isnan(f0_corr):
                                f0_corr = 0.0
                        else:
                            f0_corr = 0.0
                    else:
                        f0_corr = 0.0

                    # Output statistics (check for collapse)
                    output_std = output.std().item()

                    # Check for NaN
                    has_nan = torch.isnan(output).any().item() or torch.isnan(losses['total']).item()

                # Print header every 500 steps
                if self.global_step % 500 == 0:
                    print("\n" + "="*120)
                    print(f"{'Iter':<8} {'Loss':<8} {'GradNorm':<10} {'F0_mean':<10} {'F0_std':<10} {'F0_corr':<10} {'Out_std':<10} {'LR':<12} {'ETA':<10} {'NaN':<5}")
                    print("="*120)

                # Print metrics
                log_str = f"{self.global_step:<8} "
                log_str += f"{losses['total'].item():<8.4f} "
                log_str += f"{grad_norm:<10.3f} "
                log_str += f"{f0_mean:<10.3f} "
                log_str += f"{f0_std:<10.3f} "
                log_str += f"{f0_corr:<10.3f} "
                log_str += f"{output_std:<10.3f} "
                log_str += f"{self.optimizer.param_groups[0]['lr']:<12.6f} "
                log_str += f"{eta_seconds/60:<10.1f}m "
                log_str += f"{'YES' if has_nan else 'NO':<5}"

                # Color coding for warnings
                warnings = []
                if grad_norm < 0.1 or grad_norm > 10:
                    warnings.append(f"⚠ Grad norm: {grad_norm:.3f}")
                if abs(f0_mean) > 0.5:
                    warnings.append(f"⚠ F0 mean: {f0_mean:.3f}")
                if f0_std < 0.5 or f0_std > 1.5:
                    warnings.append(f"⚠ F0 std: {f0_std:.3f}")
                if f0_corr < 0.5:
                    warnings.append(f"⚠ F0 corr: {f0_corr:.3f}")
                if output_std < 0.2:
                    warnings.append(f"⚠ Output collapsed: {output_std:.3f}")
                if has_nan:
                    warnings.append("⚠⚠⚠ NaN DETECTED!")

                print(log_str)
                if warnings:
                    print("  " + " | ".join(warnings))

                # Store losses and metrics
                self.train_losses.append({
                    "step": self.global_step,
                    "grad_norm": grad_norm,
                    "f0_mean": f0_mean,
                    "f0_std": f0_std,
                    "f0_corr": f0_corr,
                    "output_std": output_std,
                    "has_nan": has_nan,
                    **{k: v.item() for k, v in losses.items()}
                })

            # Validation
            if self.val_loader is not None and self.global_step % self.val_interval == 0:
                val_losses = self.validate()
                print(f"Validation - Loss: {val_losses['total']:.4f}")

                # Check if best model
                if val_losses["total"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total"]
                    self.save_checkpoint("best.pt")
                    print(f"  → New best model saved!")

                self.model.train()

            # Checkpoint
            if self.global_step % self.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
                self.cleanup_checkpoints()

        print(f"\nTraining completed!")
        print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")

        # Save final checkpoint
        self.save_checkpoint("final.pt")

    def validate(self) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()

        total_losses = {}
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Adapt batch if needed (for ShardDataset compatibility)
                features, _ = _adapt_batch_if_needed(batch)
                
                features = features.to(self.device)

                # Forward pass
                output, explicit, reconstructions = self.model(
                    explicit_features=features,
                    return_reconstructions=True
                )

                # Compute loss
                losses = self.loss_fn(
                    refined=output,
                    reconstructions=reconstructions,
                    targets=features,
                )

                # Accumulate losses
                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value.item()

                num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        # Store validation losses
        self.val_losses.append({
            "step": self.global_step,
            **avg_losses
        })

        return avg_losses

    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from step {self.global_step}")

    def cleanup_checkpoints(self):
        """Keep only last N checkpoints"""
        checkpoints = sorted(
            Path(self.checkpoint_dir).glob("checkpoint_*.pt"),
            key=lambda x: int(x.stem.split("_")[1])
        )

        # Keep best and final checkpoints
        if len(checkpoints) > self.keep_last_n:
            for checkpoint in checkpoints[:-self.keep_last_n]:
                os.remove(checkpoint)


def test_trainer():
    print("Testing Prosody Trainer...")
    
    from training.dataset import collate_fn
    
    # Create dummy model
    model = ProsodyEncoder()
    
    # Create proper dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 20
        def __getitem__(self, idx):
            return torch.randn(4, 500), None
    
    train_loader = DataLoader(
        DummyDataset(),
        batch_size=4,
        collate_fn=collate_fn,
    )
    
    trainer = ProsodyTrainer(
        model=model,
        train_loader=train_loader,
        max_iterations=10,
        log_interval=5,
        device="cpu",
    )
    
    print("\nTest setup complete!")


if __name__ == "__main__":
    test_trainer()