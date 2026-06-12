"""
Train Timbre Encoder (ECAPA-TDNN)
==================================
Usage:
    python train.py --shard_dir /content/timbre_shards --speakers_per_batch 32 --num_workers 8
"""

import argparse
from pathlib import Path
import torch
import random
import os
import sys
import subprocess
import numpy as np
import yaml

_zvtvc_root = str(Path(__file__).resolve().parent.parent)
if _zvtvc_root not in sys.path:
    sys.path.insert(0, _zvtvc_root)

from Shard_dataset_unified import TimbreEncoderDataset as ShardTimbreDataset
from utils.logger import get_logger
from model.ecapa_tdnn import create_ecapa_tdnn
from training.trainer import create_trainer_from_config

logger = get_logger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def find_checkpoint(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    for name in ['best_model.pt', 'latest.pt']:
        p = ckpt_dir / name
        if p.exists():
            return str(p)
    # Find latest stage checkpoint
    stages = sorted(ckpt_dir.glob("stage*.pt"))
    if stages:
        return str(stages[-1])
    return None


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description='Train Timbre Encoder')
    p.add_argument('--shard_dir', type=str, required=True)
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--speakers_per_batch', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--skip_validate', action='store_true', help='Skip auto-validation at end')
    p.add_argument('--patience', type=int, default=None,
                   help='Early stopping patience (val checks without improvement). '
                        'Overrides config. Set 0 to disable early stopping.')
    p.add_argument('--min_delta', type=float, default=None,
                   help='Min val loss improvement to reset patience counter. Overrides config.')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config)
    tc = config.get('training', {})

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resume
    if args.resume is None:
        args.resume = find_checkpoint(ckpt_dir / "checkpoints")

    logger.info("=" * 60)
    logger.info("Timbre Encoder Training")
    logger.info("=" * 60)

    # Datasets
    shard_dir = args.shard_dir
    if not Path(shard_dir).exists():
        raise FileNotFoundError(f"Shard dir not found: {shard_dir}")

    logger.info(f"Loading from: {shard_dir}")
    train_ds = ShardTimbreDataset(shard_dir=shard_dir, split='train', is_training=True)
    val_ds = ShardTimbreDataset(shard_dir=shard_dir, split='val', is_training=False)
    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Model
    model = create_ecapa_tdnn({'input_dim': 80, 'embedding_dim': 256,
                                'input_conv_channels': 128, 'attention_channels': 128})
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} params ({n_params/1e6:.2f}M)")

    # Training config
    spk = args.speakers_per_batch
    utt = 2
    bs = spk * utt
    base_lr = tc.get('optimizer', {}).get('learning_rate', 0.001)
    total_iter = tc.get('total_iterations', 100000)

    training_config = {
        'batch': {'speakers_per_batch': spk, 'utterances_per_speaker': utt},
        'optimizer': {
            'name': 'adamw',
            'learning_rate': base_lr,
            'weight_decay': tc.get('optimizer', {}).get('weight_decay', 0.0001),
            'gradient_clip_norm': tc.get('optimizer', {}).get('gradient_clip_norm', 3.0),
        },
        'scheduler': tc.get('scheduler', {'name': 'step', 'warmup_iterations': 2000, 'min_lr': 1e-6}),
        'loss': tc.get('loss', {'temperature': 0.05}),
        'log_interval': tc.get('log_interval', 100),
        'validation_interval': tc.get('validation_interval', 2000),
        'save_checkpoint_interval': tc.get('save_checkpoint_interval', 15000),
        'total_iterations': total_iter,
        'stages': tc.get('stages', []),
        'stage_only_checkpoints': True,
        'output_dir': str(ckpt_dir),
        'use_amp': True,
        'num_workers': args.num_workers,
        'early_stopping_patience': (
            args.patience if args.patience is not None
            else tc.get('early_stopping_patience', 5)
        ),
        'early_stopping_min_delta': (
            args.min_delta if args.min_delta is not None
            else tc.get('early_stopping_min_delta', 0.001)
        ),
    }

    logger.info(f"Batch: {spk} spk x {utt} utt = {bs}")
    logger.info(f"Total iterations: {total_iter:,}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info(f"Validation every {training_config['validation_interval']} iters")
    logger.info(f"Early stopping: patience={training_config['early_stopping_patience']}")

    # Create trainer
    trainer = create_trainer_from_config(training_config, model, train_ds, val_ds)

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    try:
        trainer.train(total_iterations=total_iter)
    except KeyboardInterrupt:
        logger.warning("Training interrupted")
        trainer._save_checkpoint('interrupted.pt')
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        trainer._save_checkpoint('error.pt')
        raise

    # Auto-validate
    if not args.skip_validate:
        best_ckpt = ckpt_dir / "checkpoints" / "best_model.pt"
        if not best_ckpt.exists():
            best_ckpt = ckpt_dir / "checkpoints" / "final_model.pt"
        if best_ckpt.exists():
            logger.info("\n" + "=" * 60)
            logger.info("AUTO-VALIDATION (EER)")
            logger.info("=" * 60)
            validate_script = Path(__file__).parent / "training" / "Validate.py"
            if validate_script.exists():
                cmd = [
                    sys.executable, str(validate_script),
                    "--checkpoint", str(best_ckpt),
                    "--shard_dir", shard_dir,
                    "--model_config", args.config,
                    "--num_trials", "2000",
                    "--output_dir", str(ckpt_dir / "validation_results"),
                    "--device", args.device,
                ]
                logger.info(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd)
            else:
                logger.warning(f"Validate.py not found at {validate_script}")

    logger.info("Done!")


if __name__ == '__main__':
    main()