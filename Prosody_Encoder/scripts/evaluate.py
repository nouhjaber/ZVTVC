"""
Evaluation Script for Prosody Encoder
Uses shard-based dataset (ProsodyEncoderDataset) instead of raw audio files.
"""
import os
import sys
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add Prosody Encoder root (parent of scripts/) so model/training/utils are importable
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Add ZVTVC root (parent of Prosody Encoder/) for Shard_dataset_unified
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from model.prosody_encoder import ProsodyEncoder
from utils.metrics import ProsodyMetrics, compute_all_metrics
from Shard_dataset_unified import create_dataloader


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Evaluate Prosody Encoder")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--shard_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed shards"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of data loading workers (default: 2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print(f"\n{'='*60}")
    print(f"Prosody Encoder v{config['version']} Evaluation")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create model
    print("Creating model...")
    model = ProsodyEncoder(
        sample_rate=config["features"]["sample_rate"],
        hop_length=config["features"]["hop_length"],
        frame_rate=config["features"]["frame_rate"],
        explicit_dim=config["model"]["explicit_dim"],
        refined_dim=config["model"]["refined_dim"],
        f0_method=config["features"]["f0"]["method"],
        f0_fmin=config["features"]["f0"]["fmin"],
        f0_fmax=config["features"]["f0"]["fmax"],
        rhythm_window_size=config["features"]["rhythm"]["window_size"],
        use_refinement=True,
        output_format=config["model"].get("output_format", "refined"),
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from step {checkpoint['global_step']}")

    # Create shard-based dataloader
    print(f"\nLoading {args.split} data from shards: {args.shard_dir}")
    eval_loader = create_dataloader(
        shard_dir=args.shard_dir,
        split=args.split,
        module='prosody',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(f"Found {len(eval_loader.dataset)} samples in {args.split} split")

    if len(eval_loader.dataset) == 0:
        print("ERROR: No samples found!")
        return

    # Create metrics tracker
    metrics_tracker = ProsodyMetrics()
    num_processed = 0

    # Evaluate
    print("\nEvaluating...")
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            try:
                # Unpack batch: (features [B, 4, T], features_aug or None)
                gt_features, _ = batch
                gt_features = gt_features.to(device)

                # Forward pass
                output, explicit, reconstructions = model(
                    explicit_features=gt_features,
                    return_reconstructions=True
                )

                # Reconstruct features from output
                if reconstructions is not None:
                    pred_features = torch.cat([
                        reconstructions["f0"],
                        reconstructions["energy"],
                        reconstructions["voicing"],
                        reconstructions["rhythm"],
                    ], dim=1)

                    # Update metrics
                    metrics_tracker.update(pred_features, gt_features)

                num_processed += gt_features.shape[0]

            except Exception as e:
                print(f"\nError processing batch: {e}")
                continue

    # Compute final metrics
    print(f"\nProcessed {num_processed} samples")
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    final_metrics = metrics_tracker.compute()

    print(f"\nF0 Metrics:")
    print(f"  Pearson Correlation: {final_metrics['f0_pcc']:.4f}")
    print(f"  RMSE (normalized): {final_metrics['f0_rmse']:.4f}")

    print(f"\nEnergy Metrics:")
    print(f"  Correlation: {final_metrics['energy_correlation']:.4f}")
    print(f"  RMSE (normalized): {final_metrics['energy_rmse']:.4f}")

    print(f"\nVoicing Metrics:")
    print(f"  Error Rate: {final_metrics['voicing_error_rate']:.4f}")

    print(f"\nRhythm Metrics:")
    print(f"  Correlation: {final_metrics['rhythm_correlation']:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "evaluation_results.txt")

    with open(results_path, "w") as f:
        f.write("Prosody Encoder Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Shard dir: {args.shard_dir}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Number of samples: {num_processed}\n\n")
        f.write("Metrics:\n")
        for key, value in final_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()