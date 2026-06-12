"""
Training Script for Generator (Module 4) — FIXED VERSION

Two modes:
    # FAST: With precomputed encoder outputs (recommended)
    python train.py --shard_dir /content/outputs --use_precomputed \
        --num_workers 4 --batch_size 16

    # SLOW: With on-the-fly encoders (original)
    python train.py --shard_dir /content/shards \
        --content_encoder_ckpt ... --prosody_encoder_ckpt ... --timbre_encoder_ckpt ...

    # Resume from checkpoint (works with either mode)
    python train.py --shard_dir /content/outputs --use_precomputed --resume checkpoints/best_model.pt
"""

import torch
import argparse
import logging
from pathlib import Path

from model.unet import FlowMatchingUNet
from training.dataset import (
    PrecomputedGeneratorDataset,
    OnTheFlyGeneratorDataset,
    DummyEncoderWrapper,
    get_dataloader,
    load_content_encoder,
    load_prosody_encoder,
    load_timbre_encoder,
)
from training.trainer import GeneratorTrainer


def setup_logging(log_dir: str):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def find_best_checkpoint(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return None
    for name in ['best_model.pt', 'best.pt', 'latest.pt']:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return str(candidate)
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_iter_*.pt'))
    if checkpoints:
        return str(checkpoints[-1])
    return None


def create_model(args, device: str) -> FlowMatchingUNet:
    model = FlowMatchingUNet(
        mel_channels=args.mel_channels,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        channel_mult=list(args.channel_mult),
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        dropout=args.dropout,
        content_dim=args.content_dim,
        prosody_dim=args.prosody_dim,
        timbre_dim=args.timbre_dim,
    )
    return model.to(device)


def create_datasets_precomputed(args):
    """Create datasets from precomputed encoder outputs."""
    train_dataset = PrecomputedGeneratorDataset(
        shard_dir=args.shard_dir,
        split='train',
        lang=args.lang,
        sequence_length=args.sequence_length,
    )

    try:
        val_dataset = PrecomputedGeneratorDataset(
            shard_dir=args.shard_dir,
            split='val',
            lang=args.lang,
            sequence_length=args.sequence_length,
        )
    except ValueError:
        print("[WARNING] No val shards found, using train set for validation")
        val_dataset = train_dataset

    return train_dataset, val_dataset


def create_datasets_onthefly(args):
    """Create datasets with on-the-fly encoder inference."""
    if args.use_dummy_encoders:
        print("Using dummy encoders (for testing)")
        content_enc = DummyEncoderWrapper(args.content_dim, 'frame')
        prosody_enc = DummyEncoderWrapper(args.prosody_dim, 'frame')
        timbre_enc = DummyEncoderWrapper(args.timbre_dim, 'global')
    else:
        print("Loading real encoders from checkpoints...")
        content_enc = load_content_encoder(args.content_encoder_ckpt, args.device)
        prosody_enc = load_prosody_encoder(args.prosody_encoder_ckpt, args.device)
        timbre_enc = load_timbre_encoder(args.timbre_encoder_ckpt, args.device)

    train_dataset = OnTheFlyGeneratorDataset(
        shard_dir=args.shard_dir, split='train', lang=args.lang,
        content_encoder=content_enc, prosody_encoder=prosody_enc,
        timbre_encoder=timbre_enc,
        sequence_length=args.sequence_length, device=args.device,
    )

    try:
        val_dataset = OnTheFlyGeneratorDataset(
            shard_dir=args.shard_dir, split='val', lang=args.lang,
            content_encoder=content_enc, prosody_encoder=prosody_enc,
            timbre_encoder=timbre_enc,
            sequence_length=args.sequence_length, device=args.device,
        )
    except ValueError:
        print("[WARNING] No val shards found, using train set for validation")
        val_dataset = train_dataset

    return train_dataset, val_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train Generator (Module 4)')

    # Data
    parser.add_argument('--shard_dir', type=str, required=True,
                        help='Path to shards (/content/outputs for precomputed, /content/shards for on-the-fly)')
    parser.add_argument('--lang', type=str, default=None)

    # ---- MODE SELECTION ----
    parser.add_argument('--use_precomputed', action='store_true', default=False,
                        help='Use precomputed encoder outputs (FAST, allows num_workers>0)')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None)

    # Training
    parser.add_argument('--iterations', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sequence_length', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--ema_decay', type=float, default=0.9999)

    # Device / workers
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (>0 only with --use_precomputed)')

    # Model architecture
    parser.add_argument('--mel_channels', type=int, default=80)
    parser.add_argument('--model_channels', type=int, default=256)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 2])
    parser.add_argument('--attention_resolutions', type=int, nargs='+', default=[4])
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--content_dim', type=int, default=512)
    parser.add_argument('--prosody_dim', type=int, default=32)
    parser.add_argument('--timbre_dim', type=int, default=256)

    # On-the-fly encoder args (only needed without --use_precomputed)
    parser.add_argument('--use_dummy_encoders', action='store_true', default=False)
    parser.add_argument('--content_encoder_ckpt', type=str, default=None)
    parser.add_argument('--prosody_encoder_ckpt', type=str, default=None)
    parser.add_argument('--timbre_encoder_ckpt', type=str, default=None)

    # Logging
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=5000)
    parser.add_argument('--save_interval', type=int, default=5000)

    args = parser.parse_args()

    # Validate
    if not args.use_precomputed and not args.use_dummy_encoders:
        for name, path in [('content_encoder_ckpt', args.content_encoder_ckpt),
                           ('prosody_encoder_ckpt', args.prosody_encoder_ckpt),
                           ('timbre_encoder_ckpt', args.timbre_encoder_ckpt)]:
            if path is None:
                parser.error(f"--{name} required when not using --use_precomputed or --use_dummy_encoders")

    if args.use_precomputed and args.num_workers == 0:
        print("[INFO] --use_precomputed allows num_workers>0 for faster loading. Consider --num_workers 4")

    if not args.use_precomputed and args.num_workers > 0:
        print("[WARNING] On-the-fly mode requires num_workers=0 (GPU encoders). Forcing num_workers=0.")
        args.num_workers = 0

    return args


def main():
    args = parse_args()

    logger = setup_logging(args.checkpoint_dir)
    logger.info("=" * 70)
    logger.info("Generator (Module 4) - Training")
    logger.info("=" * 70)
    logger.info(f"Mode: {'PRECOMPUTED (fast)' if args.use_precomputed else 'ON-THE-FLY (slow)'}")
    logger.info(f"Arguments: {vars(args)}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Auto-find checkpoint
    if args.resume is None:
        args.resume = find_best_checkpoint(checkpoint_dir)
        if args.resume:
            logger.info(f"Auto-resuming from: {args.resume}")

    # Create model
    model = create_model(args, args.device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model: {num_params:.2f}M parameters")

    # Create datasets
    if args.use_precomputed:
        train_dataset, val_dataset = create_datasets_precomputed(args)
    else:
        train_dataset, val_dataset = create_datasets_onthefly(args)

    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = get_dataloader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # Build config
    config = {
        'training': {
            'total_iterations': args.iterations,
            'optimizer': {
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                'gradient_clip_norm': args.gradient_clip,
            },
            'scheduler': {
                'warmup_steps': args.warmup_steps,
                'min_lr': 1e-6,
            },
            'ema': {'enabled': True, 'decay': args.ema_decay},
            'mixed_precision': {'enabled': args.device == 'cuda'},
            'validation': {'frequency': args.val_interval},
            'logging': {'log_frequency': args.log_interval},
        },
        'checkpointing': {
            'save_frequency': args.save_interval,
            'keep_last_n': 5,
        },
    }

    # Create trainer
    trainer = GeneratorTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        logger=logger,
    )

    # Resume
    if args.resume:
        logger.info(f"Loading checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Training Configuration")
    logger.info("=" * 70)
    logger.info(f"  Mode:        {'PRECOMPUTED' if args.use_precomputed else 'ON-THE-FLY'}")
    logger.info(f"  Iterations:  {args.iterations}")
    logger.info(f"  Batch size:  {args.batch_size}")
    logger.info(f"  Seq length:  {args.sequence_length}")
    logger.info(f"  LR:          {args.lr}")
    logger.info(f"  Warmup:      {args.warmup_steps}")
    logger.info(f"  Workers:     {args.num_workers}")
    logger.info("=" * 70 + "\n")

    trainer.train(total_iterations=args.iterations)

    logger.info("Training completed!")


if __name__ == '__main__':
    main()