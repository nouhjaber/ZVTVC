"""
Train Prosody Encoder
=====================
Usage:
    python train.py --shard_dir ../shards
    python train.py --shard_dir ../shards --resume checkpoints/best.pt
"""

import os
import argparse
import random
import time
import torch
import numpy as np
from pathlib import Path
import sys

# Add ZVTVC root to path
zvtvc_root = Path(__file__).parent.parent
if str(zvtvc_root) not in sys.path:
    sys.path.insert(0, str(zvtvc_root))

from Shard_dataset_unified import create_dataloader

from model.prosody_encoder import ProsodyEncoder
from training.trainer import ProsodyTrainer
from training.losses import ProsodyLoss


# =============================================================================
# AUTO-CALCULATION FUNCTIONS
# =============================================================================

def calculate_learning_rate(base_lr: float, batch_size: int, base_batch_size: int = 16) -> float:
    """
    Scale LR linearly with batch size.
    Reference: batch=16, lr=1e-4
    """
    scale_factor = batch_size / base_batch_size
    scaled_lr = base_lr * scale_factor
    print(f"[LR] {base_lr:.6f} × {scale_factor:.2f} = {scaled_lr:.6f}  (batch {base_batch_size}→{batch_size})")
    return scaled_lr


def calculate_total_iterations(batch_size: int, base_batch_size: int = 16,
                               base_iterations: int = 50000) -> int:
    """
    Scale iterations inversely with batch size.
    """
    scale_factor = base_batch_size / batch_size
    total = int(base_iterations * scale_factor)
    print(f"[Iterations] batch {base_batch_size}→{batch_size}  total={total:,}")
    return total


def calculate_warmup_iterations(total_iterations: int, warmup_ratio: float = 0.02) -> int:
    """Warmup = 2% of total iterations, minimum 500."""
    warmup = max(500, int(total_iterations * warmup_ratio))
    print(f"[Warmup] {warmup:,} iterations ({warmup_ratio*100:.0f}% of {total_iterations:,})")
    return warmup


# =============================================================================
# HELPERS
# =============================================================================

def find_best_checkpoint(checkpoint_dir: Path):
    """Find best/latest checkpoint in directory."""
    if not checkpoint_dir.exists():
        return None
    for name in ['best.pt', 'final.pt']:
        p = checkpoint_dir / name
        if p.exists():
            print(f"Found checkpoint: {p}")
            return str(p)
    checkpoints = list(checkpoint_dir.glob('checkpoint_*.pt'))
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[1]))
        print(f"Found latest checkpoint: {checkpoints[-1]}")
        return str(checkpoints[-1])
    return None


# =============================================================================
# ARGS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Prosody Encoder')

    # Data
    parser.add_argument('--shard_dir', type=str, default=None,
                        help='Path to preprocessed shards (default: ../shards)')

    # Training
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from (default: auto-find)')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Total iterations (default: auto from batch size)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: 16)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sequence_length', type=int, default=200,
                        help='Sequence length in frames (default: 200)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    # Model
    parser.add_argument('--refined_dim', type=int, default=32,
                        help='Prosody refined dimension (default: 32)')

    # Optimizer
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: auto from batch size)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_clip', type=float, default=1.0)

    # Loss weights
    parser.add_argument('--reconstruction_weight', type=float, default=1.0)
    parser.add_argument('--consistency_weight',    type=float, default=0.5)
    parser.add_argument('--smoothness_weight',     type=float, default=0.1)

    # Logging
    parser.add_argument('--log_interval',  type=int, default=100)
    parser.add_argument('--val_interval',  type=int, default=2000)
    parser.add_argument('--save_interval', type=int, default=5000)

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU")
        args.device = 'cpu'

    print("=" * 80)
    print("Prosody Encoder Training")
    print("=" * 80)

    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir        = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = ProsodyEncoder(refined_dim=args.refined_dim)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # FIX: resolve batch_size before using it
    if args.batch_size is None:
        args.batch_size = 16  # matches base_batch_size reference
        print(f"batch_size not set — defaulting to {args.batch_size}")

    # =================================================================
    # AUTO-CALCULATE TRAINING PARAMETERS
    # =================================================================
    print("\n" + "=" * 80)
    print("CALCULATING TRAINING PARAMETERS")
    print("=" * 80)

    base_batch_size = 16
    base_lr         = 1e-4

    scaled_lr = calculate_learning_rate(base_lr, args.batch_size, base_batch_size) \
                if args.lr is None else args.lr
    if args.lr is not None:
        print(f"[LR] Manual: {scaled_lr:.6f}")

    total_iterations = calculate_total_iterations(args.batch_size, base_batch_size, 50000) \
                       if args.iterations is None else args.iterations
    if args.iterations is not None:
        print(f"[Iterations] Manual: {total_iterations:,}")

    warmup_iterations = calculate_warmup_iterations(total_iterations, warmup_ratio=0.02)

    print(f"\n[Summary]")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Total iterations: {total_iterations:,}")
    print(f"  Learning rate:    {scaled_lr:.6f}")
    print(f"  Warmup:           {warmup_iterations:,}")
    print("=" * 80 + "\n")

    # =================================================================
    # DATALOADERS
    # =================================================================
    print("Creating datasets...")

    shard_dir = args.shard_dir or str(Path(__file__).parent.parent / "shards")
    if not Path(shard_dir).exists():
        raise FileNotFoundError(f"Shard directory not found: {shard_dir}")
    print(f"Loading from shards: {shard_dir}")

    train_loader = create_dataloader(
        shard_dir=shard_dir,
        split='train',
        module='prosody',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
        use_augmentation=True,
    )
    val_loader = create_dataloader(
        shard_dir=shard_dir,
        split='val',
        module='prosody',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
        use_augmentation=False,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # =================================================================
    # TRAINER
    # =================================================================
    loss_fn = ProsodyLoss(
        reconstruction_weight=args.reconstruction_weight,
        consistency_weight=args.consistency_weight,
        smoothness_weight=args.smoothness_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=args.weight_decay,
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iterations,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iterations - warmup_iterations, eta_min=scaled_lr * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iterations],
    )

    print("\nCreating trainer...")
    trainer = ProsodyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        max_iterations=total_iterations,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        checkpoint_interval=args.save_interval,
        grad_clip_norm=args.gradient_clip,
    )

    # Auto-resume
    if args.resume is None:
        args.resume = find_best_checkpoint(checkpoint_dir)
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Config summary
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Total iterations: {total_iterations:,}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Sequence length:  {args.sequence_length}")
    print(f"Learning rate:    {scaled_lr:.6f}")
    print(f"Warmup:           {warmup_iterations:,}")
    print(f"Weight decay:     {args.weight_decay}")
    print(f"Gradient clip:    {args.gradient_clip}")
    print(f"Device:           {args.device}  Workers: {args.num_workers}")
    print("=" * 60 + "\n")

    # Train
    print("Starting training loop...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint('interrupted.pt')
        print("Checkpoint saved")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()